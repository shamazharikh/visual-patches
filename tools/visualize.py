#!/usr/bin/env python3
"""Render what vpatch extracted from one frame, as a self-contained HTML page.

    uv run python tools/visualize.py CLIP.mp4 -o out.html [--frame N] [--cell 16]

Six panels over the same frame: the decoded picture, the encoder's own partition
geometry, its motion field, its per-macroblock QP, the aggregated grid tokens, and which
of those tokens survive anchored pruning. The point is to make the claims checkable by
eye -- block sizes really are only four, motion really does concentrate where things
move, and the variance channel really does light up where a mean would not.

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
from vpatch.types import UnitKind

IDX = {n: i for i, n in enumerate(FEATURE_LAYOUT)}

# Colourblind-safe, and ordered so finer partitions read as hotter.
SHAPE_COLOURS = {
    (16, 16): "#3b6ea5",
    (16, 8): "#4fa3a5",
    (8, 16): "#d9a441",
    (8, 8): "#c8553d",
}
FILL_COLOUR = "#6b6b6b"


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
    return "rgb(%d,%d,%d)" % tuple(int(a[k] + (b[k] - a[k]) * f) for k in range(3))


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

    panels = [
        ("Decoded frame",
         f"{w}x{h}, {frame.pict_type}-frame, display index {frame_index} of {len(frames)}",
         _svg(w, h, "", img)),
        ("Coding units the encoder chose",
         f"{len(observed)} observed partitions; "
         + ", ".join(f"{k[0]}x{k[1]}: {v}" for k, v in sorted(shapes.items(), reverse=True))
         + f"; {len(frame.units) - len(observed)} grid-filled",
         _svg(w, h, panel_partitions(frame), img, dim=0.55)),
        ("Motion field",
         f"{len(mv_units)} units carry motion, peak {max_speed:.1f}px; arrows drawn 3x",
         _svg(w, h, panel_motion(frame), img, dim=0.62)),
        ("Per-macroblock QP",
         (f"range {int(frame.qp_map.min())}-{int(frame.qp_map.max())} across "
          f"{frame.qp_map.shape[1]}x{frame.qp_map.shape[0]} macroblocks"
          if frame.qp_map is not None else "not exported by this codec"),
         _svg(w, h, panel_qp(frame), img, dim=0.4)),
        (f"Grid tokens - motion magnitude (cell {cell})",
         f"{rows}x{cols} = {rows * cols} tokens for this frame",
         _svg(w, h, panel_cells(feat[..., IDX["l0_mag_mean"]], cell, w, h,
                                max(max_speed, 1.0)), img, dim=0.5)),
        ("Grid tokens - motion variance",
         f"{int((std_ch > 0.5).sum())} cells where the encoder found more than one "
         "motion inside one cell; a plain mean over the cell would erase exactly this",
         _svg(w, h, panel_cells(std_ch, cell, w, h, max_std), img, dim=0.5)),
        ("Survives anchored pruning",
         f"{int(kept.sum())} of {kept.size} kept on this frame; clip kept "
         f"{report.kept_fraction * 100:.1f}%, cells never emitted: {report.cells_never_kept}",
         _svg(w, h, panel_kept(kept, rows, cols, cell, w, h), img, dim=0.45)),
    ]

    legend = "".join(
        f'<span class="k"><i style="background:{c}"></i>{s[0]}x{s[1]}</span>'
        for s, c in SHAPE_COLOURS.items()
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
from the decoded picture. Source: <code>{html.escape(path.rsplit("/", 1)[-1])}</code>,
{html.escape(cap.codec)}.</p>
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
                    help="display index; default picks the frame with the most motion")
    ap.add_argument("--cell", type=int, default=16)
    args = ap.parse_args()
    print(build(args.path, args.out, args.frame, args.cell))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
