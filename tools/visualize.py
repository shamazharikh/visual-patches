#!/usr/bin/env python3
"""Render what vpatch extracted from one frame, as a self-contained HTML page.

    uv run python tools/visualize.py CLIP.mp4 -o out.html [--frame N] [--cell 16]
    uv run python tools/visualize.py PHOTO.jpg -o out.html

Eight panels over the same frame: the decoded picture, the encoder's own partition
geometry, its motion field, its per-macroblock and per-unit QP, the aggregated grid
tokens, and which of those tokens survive anchored pruning. The point is to make the
claims checkable by eye -- block sizes really are only four, motion really does
concentrate where things move, and the variance channel really does light up where a
mean would not.

A panel with nothing behind it is drawn as an explicit empty card naming the reason,
never as a blank overlay with a caption implying data. Two reasons are distinguished,
because they are not the same fact: the codec exports nothing (HEVC, AV1, VP8, and the
still formats -- JPEG, PNG, WebP), or the input is a single image, where a motion vector
has no reference frame to point at and pruning has nothing to prune against. A still
therefore renders the picture, the fallback grid vpatch synthesises over it, and little
else -- which is the honest answer, not a degraded one.

Writes one file with the image embedded, so it can be opened anywhere. No plotting
dependency: the photo layer is JPEG-encoded through PyAV and the overlays are hand-built
SVG.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import math

import av
import numpy as np

from vpatch.backends.ffmpeg_video import VideoExtractor
from vpatch.patchify import FEATURE_LAYOUT, patchify_grid
from vpatch.sampling import anchor_delta, keyframe_anchors

IDX = {n: i for i, n in enumerate(FEATURE_LAYOUT)}

# Hues taken from SMPTE colour bars -- the canonical video test signal -- and desaturated
# so they work as UI. Keyed by block size rather than by codec, so a 16x16 reads the same
# whether it came from an H.264 macroblock or a VP9 leaf.
SHAPE_COLOURS = {
    (64, 64): "#9c5195",
    (32, 32): "#3a6ea8",
    (16, 16): "#2f8f96",
    (16, 8): "#c9a227",
    (8, 16): "#3f8f5c",
    (8, 8): "#c04a3d",
}
FILL_COLOUR = "#6b7480"


def _jpeg_data_uri(luma: np.ndarray, quality: int = 85) -> str:
    """Encode a luma plane to a JPEG data URI using PyAV, so there is no PIL dependency."""
    h, w = luma.shape
    rgb = np.repeat(luma[:, :, None], 3, axis=2)
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mjpeg") as container:
        stream = container.add_stream("mjpeg", rate=1)
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuvj420p"
        stream.codec_context.qmin = stream.codec_context.qmax = max(
            2, int(31 - quality * 0.29)
        )
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _svg(w: int, h: int, body: str, image: str | None = None, dim: float = 0.0) -> str:
    layers = []
    if image:
        layers.append(f'<image href="{image}" x="0" y="0" width="{w}" height="{h}"/>')
        if dim:
            layers.append(f'<rect width="{w}" height="{h}" fill="#0b0d10" '
                          f'opacity="{dim}"/>')
    layers.append(body)
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'preserveAspectRatio="xMidYMid meet" role="img">{"".join(layers)}</svg>')


def panel_partitions(frame) -> str:
    """One rectangle per coding unit, coloured by shape. Grid fill is drawn muted."""
    parts = []
    for u in frame.units:
        colour = (SHAPE_COLOURS.get((u.w, u.h), "#8a8a8a")
                  if u.geometry_observed else FILL_COLOUR)
        opacity = 0.85 if u.geometry_observed else 0.28
        parts.append(
            f'<rect x="{u.x}" y="{u.y}" width="{u.w}" height="{u.h}" fill="none" '
            f'stroke="{colour}" stroke-width="0.9" opacity="{opacity}"/>'
        )
    return "".join(parts)


def panel_motion(frame, scale: float = 3.0, min_mag: float = 0.35) -> str:
    """One arrow per unit, from the block centre, drawn at `scale` x true length."""
    parts = []
    for u in frame.units:
        if not u.mvs:
            continue
        dx = sum(m.dx for m in u.mvs) / len(u.mvs)
        dy = sum(m.dy for m in u.mvs) / len(u.mvs)
        mag = math.hypot(dx, dy)
        if mag < min_mag:
            continue
        cx, cy = u.x + u.w / 2, u.y + u.h / 2
        ex, ey = cx + dx * scale, cy + dy * scale
        # Warm for fast, cool for slow; capped so one outlier does not flatten the rest.
        t = min(mag / 8.0, 1.0)
        colour = f"rgb({int(60 + 195 * t)},{int(160 - 60 * t)},{int(220 - 160 * t)})"
        parts.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
            f'stroke="{colour}" stroke-width="{0.8 + 1.4 * t:.2f}" opacity="0.95"/>'
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="{0.7 + t:.2f}" fill="{colour}"/>'
        )
    return "".join(parts)


def _heat(v: float) -> str:
    """Perceptually ordered dark->warm ramp for a value already normalised to [0, 1]."""
    v = max(0.0, min(1.0, v))
    stops = [(13, 20, 33), (32, 74, 116), (72, 150, 148), (214, 176, 84), (206, 78, 58)]
    pos = v * (len(stops) - 1)
    i = min(int(pos), len(stops) - 2)
    f = pos - i
    a, b = stops[i], stops[i + 1]
    r, g, bl = (int(a[k] + (b[k] - a[k]) * f) for k in range(3))
    return f"rgb({r},{g},{bl})"


def panel_qp(frame) -> str:
    if frame.qp_map is None:
        return '<text x="8" y="20" fill="#888" font-size="14">no per-block QP</text>'
    qp = frame.qp_map.astype(np.float32)
    lo, hi = float(qp.min()), float(qp.max())
    span = max(hi - lo, 1e-6)
    parts = []
    for r in range(qp.shape[0]):
        for c in range(qp.shape[1]):
            parts.append(
                f'<rect x="{c * 16}" y="{r * 16}" width="16" height="16" '
                f'fill="{_heat((qp[r, c] - lo) / span)}" opacity="0.82"/>'
            )
    return "".join(parts)


def panel_unit_qp(frame) -> str:
    """Shade each coding unit by its own QP.

    Distinct from the macroblock QP map: VP9 carries `delta_qp` per block of a variable
    tree, so there is no fixed grid to draw it on. This is the only way to see it.
    """
    qps = [u.qp for u in frame.units if u.qp is not None]
    if not qps:
        return '<text x="8" y="20" fill="#888" font-size="14">no per-unit QP</text>'
    lo, hi = min(qps), max(qps)
    span = max(hi - lo, 1e-6)
    parts = []
    for u in frame.units:
        if u.qp is None:
            continue
        parts.append(
            f'<rect x="{u.x}" y="{u.y}" width="{u.w}" height="{u.h}" '
            f'fill="{_heat((u.qp - lo) / span)}" opacity="0.8" '
            f'stroke="#07090c" stroke-width="0.4"/>'
        )
    return "".join(parts)


def panel_cells(values: np.ndarray, cell: int, w: int, h: int, norm: float) -> str:
    """One square per grid token, shaded by a [rows, cols] value array."""
    rows, cols = values.shape
    parts = []
    for r in range(rows):
        for c in range(cols):
            v = float(values[r, c]) / norm
            if v <= 0.01:
                continue
            x, y = c * cell, r * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{min(cell, w - x)}" '
                f'height="{min(cell, h - y)}" fill="{_heat(v)}" opacity="0.8"/>'
            )
    return "".join(parts)


def panel_kept(kept_mask, rows, cols, cell, w, h) -> str:
    """Green where a token survives pruning, faint red where it was dropped."""
    parts = []
    for r in range(rows):
        for c in range(cols):
            x, y = c * cell, r * cell
            live = bool(kept_mask[r, c])
            parts.append(
                f'<rect x="{x}" y="{y}" width="{min(cell, w - x)}" '
                f'height="{min(cell, h - y)}" '
                f'fill="{"#4c9a5a" if live else "#8c3b34"}" '
                f'opacity="{0.72 if live else 0.2}"/>'
            )
    return "".join(parts)


def _empty(w: int, h: int, reason: str) -> str:
    """A panel with no data behind it, drawn so it cannot be mistaken for a null result.

    The alternative -- an overlay that simply draws nothing -- is worse than useless
    here, because the caption underneath it would still report a number. Zero motion
    because the codec exports none and zero motion because the input is one photograph
    are different facts, and neither is "the encoder found no motion".
    """
    # Drawn as a short band rather than at frame height: a still leaves six of the eight
    # panels empty, and six full-size black rectangles would make the page mostly a
    # scroll through nothing. Width is kept so the cards still line up.
    band = max(44.0, h / 5)
    return (f'<svg viewBox="0 0 {w} {band:.0f}" xmlns="http://www.w3.org/2000/svg" '
            f'preserveAspectRatio="xMidYMid meet" role="img">'
            f'<rect x="1" y="1" width="{w - 2}" height="{band - 2:.0f}" fill="#0b0d10" '
            f'stroke="#39414c" stroke-width="1.2" stroke-dasharray="7 6" rx="6"/>'
            f'<text x="{w / 2}" y="{band / 2:.0f}" fill="#7f8b99" '
            f'font-size="{band / 3.2:.1f}" dominant-baseline="central" '
            f'font-family="ui-sans-serif,system-ui,sans-serif" text-anchor="middle">'
            f'{html.escape(reason)}</text></svg>')


def build(path: str, out: str, frame_index: int | None, cell: int) -> str:
    ex = VideoExtractor(path, pixels=True, max_frames=64)
    cap = ex.capability()
    frames = ex.extract()
    frames.sort(key=lambda f: f.display_index)

    bundle = patchify_grid(frames, cell=cell)
    anchors = keyframe_anchors(frames)
    _, report = anchor_delta(bundle, anchors=anchors)

    speed = np.hypot(bundle.features[:, IDX["l0_dx_mean"]],
                     bundle.features[:, IDX["l0_dy_mean"]])
    if frame_index is None:
        # argmax over an all-zero motion array silently returns 0; that is the right
        # frame for a still and an arbitrary one for a codec that exports no motion,
        # so only the first case gets to call itself a choice.
        per_frame = [float(speed[bundle.times == t].sum()) for t in range(len(frames))]
        frame_index = int(np.argmax(per_frame))
    frame = frames[frame_index]

    from vpatch.patchify import frame_features
    feat, _, _ = frame_features(frame, cell)
    rows, cols = feat.shape[:2]
    w, h = frame.width, frame.height

    # Recompute the keep mask for this one frame, the same way sampling does.
    sel = bundle.times == frame_index
    f = bundle.features[sel]
    sp = np.maximum(np.hypot(f[:, IDX["l0_dx_mean"]], f[:, IDX["l0_dy_mean"]]),
                    np.hypot(f[:, IDX["l1_dx_mean"]], f[:, IDX["l1_dy_mean"]]))
    sd = np.maximum(np.hypot(f[:, IDX["l0_dx_std"]], f[:, IDX["l0_dy_std"]]),
                    np.hypot(f[:, IDX["l1_dx_std"]], f[:, IDX["l1_dy_std"]]))
    kept = ((sp >= 0.5) | (sd >= 0.5) | (frame_index in anchors)).reshape(rows, cols)

    img = _jpeg_data_uri(frame.pixels)
    observed = [u for u in frame.units if u.geometry_observed]
    shapes: dict[tuple[int, int], int] = {}
    for u in observed:
        shapes[(u.w, u.h)] = shapes.get((u.w, u.h), 0) + 1
    mv_units = [u for u in observed if u.mvs]
    speeds = [math.hypot(sum(m.dx for m in u.mvs) / len(u.mvs),
                         sum(m.dy for m in u.mvs) / len(u.mvs)) for u in mv_units]

    max_speed = max(speeds) if speeds else 1.0
    std_ch = np.hypot(feat[..., IDX["l0_dx_std"]], feat[..., IDX["l0_dy_std"]])
    max_std = float(std_ch.max()) or 1.0

    # A single decoded frame is a still: not a short clip, a different question. Motion
    # vectors are displacements from a reference picture and pruning is defined against
    # a previous frame, so both are undefined here rather than empty. Keyed on the frame
    # count, not the file extension -- a one-frame .mp4 is a still, and an animated GIF
    # is not.
    still = len(frames) == 1
    no_motion = "a single image has no reference frame to measure motion against" \
        if still else f"{cap.codec} exports no motion vectors at any ffmpeg version"
    has_motion = bool(mv_units)
    unit_qps = [u.qp for u in frame.units if u.qp is not None]

    if observed:
        units_sub = (f"{len(observed)} observed partitions; "
                     + ", ".join(f"{k[0]}x{k[1]}: {v}"
                                 for k, v in sorted(shapes.items(), reverse=True))
                     + f"; {len(frame.units) - len(observed)} grid-filled")
    else:
        # Every unit is synthetic. Worth drawing anyway -- it is exactly what patchify
        # aggregates over -- but the title must not credit the encoder for it.
        units_sub = (f"none exported by {cap.codec}; all {len(frame.units)} units are the "
                     f"fallback 16x16 grid vpatch synthesises to keep the layout defined")

    panels = [
        ("Decoded frame",
         (f"{w}x{h}, {frame.pict_type}-frame, single image"
          if still else
          f"{w}x{h}, {frame.pict_type}-frame, display index {frame_index} "
          f"of {len(frames)}"),
         _svg(w, h, "", img)),
        ("Coding units the encoder chose" if observed else "Coding units - none exported",
         units_sub,
         _svg(w, h, panel_partitions(frame), img, dim=0.55)),
        ("Motion field",
         (f"{len(mv_units)} units carry motion, peak {max_speed:.1f}px; arrows drawn 3x"
          if has_motion else no_motion),
         _svg(w, h, panel_motion(frame), img, dim=0.62) if has_motion
         else _empty(w, h, "no motion vectors")),
        ("Per-macroblock QP",
         (f"range {int(frame.qp_map.min())}-{int(frame.qp_map.max())} across "
          f"{frame.qp_map.shape[1]}x{frame.qp_map.shape[0]} macroblocks"
          if frame.qp_map is not None else f"not exported by {cap.codec}"),
         _svg(w, h, panel_qp(frame), img, dim=0.4) if frame.qp_map is not None
         else _empty(w, h, "no per-macroblock QP")),
        ("Per-unit QP",
         (f"{len(unit_qps)} units carry their own QP, range {min(unit_qps)}-"
          f"{max(unit_qps)}"
          + ("; the same values the map above shows, on the coding-unit grid"
             if frame.qp_map is not None else
             "; VP9 carries this per block of a variable tree, so there is no fixed "
             "grid to draw it on and the panel above cannot show it")
          if unit_qps else f"not exported by {cap.codec}"),
         _svg(w, h, panel_unit_qp(frame), img, dim=0.4) if unit_qps
         else _empty(w, h, "no per-unit QP")),
        (f"Grid tokens - motion magnitude (cell {cell})",
         (f"{rows}x{cols} = {rows * cols} tokens for this frame"
          if has_motion else
          f"{rows}x{cols} = {rows * cols} tokens are still laid out, but the motion "
          f"channels are all zero: {no_motion}"),
         _svg(w, h, panel_cells(feat[..., IDX["l0_mag_mean"]], cell, w, h,
                                max(max_speed, 1.0)), img, dim=0.5) if has_motion
         else _empty(w, h, "no motion to aggregate")),
        ("Grid tokens - motion variance",
         (f"{int((std_ch > 0.5).sum())} cells where the encoder found more than one "
          "motion inside one cell; a plain mean over the cell would erase exactly this"
          if has_motion else no_motion),
         _svg(w, h, panel_cells(std_ch, cell, w, h, max_std), img, dim=0.5) if has_motion
         else _empty(w, h, "no motion to aggregate")),
        ("Survives anchored pruning",
         (f"{int(kept.sum())} of {kept.size} kept on this frame; clip kept "
          f"{report.kept_fraction * 100:.1f}%, cells never emitted: "
          f"{report.cells_never_kept}"
          if not still else
          "undefined for a still: the only frame is itself the anchor, so every token "
          "survives by construction and the 100% would measure nothing"),
         _svg(w, h, panel_kept(kept, rows, cols, cell, w, h), img, dim=0.45)
         if not still else _empty(w, h, "nothing to prune against")),
    ]

    # The shape swatches are a key to colours that only appear when the encoder exported
    # geometry; on a still they would advertise six block sizes the page never draws.
    legend = (
        "".join(f'<span class="k"><i style="background:{c}"></i>{s[0]}x{s[1]}</span>'
                for s, c in SHAPE_COLOURS.items()) if observed else ""
    ) + f'<span class="k"><i style="background:{FILL_COLOUR}"></i>grid fill</span>'

    cards = "".join(
        f'<figure><h2>{html.escape(t)}</h2><p>{html.escape(sub)}</p>'
        f'<div class="frame">{svg}</div></figure>'
        for t, sub, svg in panels
    )

    doc = f"""<title>vpatch panels</title>
<style>
  :root {{ --bg:#f6f7f9; --fg:#14181d; --mut:#5c6672; --card:#fff; --line:#e2e6eb; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
    --bg:#0f1216; --fg:#e8ecf1; --mut:#9aa5b1; --card:#171b21; --line:#262c34; }} }}
  :root[data-theme=dark] {{ --bg:#0f1216; --fg:#e8ecf1; --mut:#9aa5b1;
    --card:#171b21; --line:#262c34; }}
  body {{ margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
    font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
  main {{ max-width:1180px; margin:0 auto; }}
  h1 {{ font-size:1.5rem; margin:0 0 .3rem; letter-spacing:-.01em; }}
  .lead {{ color:var(--mut); margin:0 0 1.5rem; max-width:62ch; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:.85rem; margin:0 0 1.5rem;
    font-size:.82rem; color:var(--mut); }}
  .k {{ display:inline-flex; align-items:center; gap:.4rem; }}
  .k i {{ width:.72rem; height:.72rem; border-radius:2px; display:inline-block; }}
  figure {{ margin:0 0 1.6rem; background:var(--card); border:1px solid var(--line);
    border-radius:10px; padding:1rem 1rem 1.1rem; }}
  h2 {{ font-size:.95rem; margin:0 0 .2rem; }}
  figure p {{ margin:0 0 .7rem; color:var(--mut); font-size:.82rem; }}
  .frame {{ overflow-x:auto; }}
  svg {{ width:100%; height:auto; display:block; border-radius:6px;
    background:#0b0d10; }}
  footer {{ color:var(--mut); font-size:.8rem; margin-top:2rem; }}
  code {{ font-size:.85em; }}
</style>
<main>
<h1>What the codec already knew</h1>
<p class="lead">Every overlay below is read out of the compressed bitstream, not computed
from the decoded picture -- so where the bitstream carries nothing, the panel says so
instead of drawing an empty one. Source:
<code>{html.escape(path.rsplit("/", 1)[-1])}</code>, {html.escape(cap.codec)}{
" -- a single image" if still else f", {len(frames)} frames"}.</p>
<div class="legend">{legend}</div>
{cards}
<footer>Generated by <code>tools/visualize.py</code>. Motion is in luma pixels; no
reference index is exported, so vectors are not normalised by temporal distance.</footer>
</main>"""
    with open(out, "w") as fh:
        fh.write(doc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-o", "--out", default="vpatch-panels.html")
    ap.add_argument("--frame", type=int, default=None,
                    help="display index; default picks the frame with the most motion, "
                         "or frame 0 for a still or a codec that exports no motion")
    ap.add_argument("--cell", type=int, default=16)
    args = ap.parse_args()
    print(build(args.path, args.out, args.frame, args.cell))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
