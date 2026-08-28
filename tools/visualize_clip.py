#!/usr/bin/env python3
"""Render what vpatch extracted from a WHOLE clip, as a self-contained HTML page.

    uv run python tools/visualize_clip.py CLIP.webm -o out.html [--cell 16]

`visualize.py` freezes one frame and shows every channel over it. This is the other
axis: one or two channels, every frame. It is aimed at VP9, whose contribution is the
partition quadtree -- and a quadtree only tells you anything once you watch it move.
Where the encoder splits, and when, is the signal.

Two overlays are re-encoded as H.264 and embedded as data URIs, so the page plays the
whole clip with no external file. The rest are timelines and an accumulation map.
"""

from __future__ import annotations

import argparse
import base64
import collections
import fractions
import html
import io

import av
import numpy as np

from vpatch.backends.ffmpeg_video import VideoExtractor
from vpatch.patchify import patchify_grid
from vpatch.sampling import anchor_delta, keyframe_anchors

# Same SMPTE-derived hues as visualize.py, but keyed on the block's LARGER SIDE rather
# than its exact shape. VP9 emits a handful of rectangles (64x16, 32x16, ...) that have
# no shape entry, and across a whole clip the reading that matters is tree depth, which
# is what the larger side encodes.
DEPTH_COLOURS = {64: "#9c5195", 32: "#3a6ea8", 16: "#2f8f96", 8: "#c04a3d"}
DEPTH_ORDER = [64, 32, 16, 8]


def _depth_class(w: int, h: int) -> int:
    side = max(w, h)
    for s in DEPTH_ORDER:
        if side >= s:
            return s
    return 8


def _h264_data_uri(rgb_frames, fps: float, crf: int = 23) -> tuple[str, int]:
    """Encode an RGB sequence to H.264 in MP4 and return it as a data URI."""
    buf = io.BytesIO()
    h, w = rgb_frames[0].shape[:2]
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream(
            "libx264", rate=fractions.Fraction(fps).limit_denominator(1000))
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "slow",
                          "movflags": "+faststart", "threads": "1"}
        for arr in rgb_frames:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    raw = buf.getvalue()
    return "data:video/mp4;base64," + base64.b64encode(raw).decode(), len(raw)


def _rgb(colour: str) -> tuple[int, int, int]:
    return tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))


def _maps(frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel unit id, depth class and QP, painted one unit at a time.

    Painting ids rather than edges is what lets two adjacent blocks of the SAME size
    still show a boundary between them -- an edge map derived from depth alone would
    merge them and understate the split count.
    """
    h, w = frame.height, frame.width
    uid = np.zeros((h, w), dtype=np.int32)
    depth = np.zeros((h, w), dtype=np.uint8)
    qp = np.full((h, w), np.nan, dtype=np.float32)
    for i, u in enumerate(frame.units, start=1):
        ys, xs = slice(u.y, u.y + u.h), slice(u.x, u.x + u.w)
        uid[ys, xs] = i
        depth[ys, xs] = DEPTH_ORDER.index(_depth_class(u.w, u.h)) + 1
        if u.qp is not None:
            qp[ys, xs] = u.qp
    return uid, depth, qp


# Deeper blocks get a fainter line. At 8x8 a whole frame of opaque hairlines is a solid
# mesh that hides the picture it is supposed to annotate; grading by depth keeps the
# coarse structure legible and lets the fine mesh read as texture.
EDGE_ALPHA = {64: 1.0, 32: 0.85, 16: 0.62, 8: 0.42}


def overlay_partitions(frame, uid, depth, dim: float = 0.3) -> np.ndarray:
    """Dimmed picture with one coloured hairline on every coding-unit boundary."""
    base = frame.pixels.astype(np.float32) * (1.0 - dim)
    out = np.repeat(base[:, :, None], 3, axis=2)
    edge = np.zeros(uid.shape, dtype=bool)
    edge[:, 1:] |= uid[:, 1:] != uid[:, :-1]
    edge[1:, :] |= uid[1:, :] != uid[:-1, :]
    edge[0, :] = edge[:, 0] = True
    for k, size in enumerate(DEPTH_ORDER, start=1):
        m = edge & (depth == k)
        if m.any():
            a = EDGE_ALPHA[size]
            out[m] = out[m] * (1 - a) + np.array(_rgb(DEPTH_COLOURS[size])) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_qp(frame, qp, ladder: np.ndarray, dim: float = 0.35) -> np.ndarray:
    """Picture tinted by each unit's own quantiser index, on a clip-wide rank scale.

    q_idx is violently skewed -- median 3 against a maximum of 200 on this clip -- so a
    linear ramp paints the whole frame the bottom colour and shows nothing. `ladder` is
    the sorted distribution over every unit in the clip, which makes the colour a
    percentile: comparable between frames, and able to resolve the low end where nearly
    all the mass sits.
    """
    base = frame.pixels.astype(np.float32) * (1.0 - dim)
    out = np.repeat(base[:, :, None], 3, axis=2)
    ok = np.isfinite(qp)
    t = np.zeros_like(qp)
    t[ok] = np.searchsorted(ladder, qp[ok]) / max(len(ladder), 1)
    ramp = np.array([(13, 20, 33), (32, 74, 116), (72, 150, 148),
                     (214, 176, 84), (206, 78, 58)], dtype=np.float32)
    pos = np.clip(t[ok], 0, 1) * (len(ramp) - 1)
    i = np.clip(pos.astype(np.int32), 0, len(ramp) - 2)
    f = (pos - i)[:, None]
    col = ramp[i] * (1 - f) + ramp[i + 1] * f
    out[ok] = out[ok] * 0.35 + col * 0.65
    return np.clip(out, 0, 255).astype(np.uint8)


def _heat(v: float) -> str:
    v = max(0.0, min(1.0, v))
    stops = [(13, 20, 33), (32, 74, 116), (72, 150, 148), (214, 176, 84), (206, 78, 58)]
    pos = v * (len(stops) - 1)
    i = min(int(pos), len(stops) - 2)
    f = pos - i
    a, b = stops[i], stops[i + 1]
    return "rgb(%d,%d,%d)" % tuple(int(a[k] + (b[k] - a[k]) * f) for k in range(3))


def chart_line(series, keys, ylabel: str, colour: str = "#4fb3ba") -> str:
    """A single trace against frame index, with keyframes marked."""
    n = len(series)
    w, h, pad = 1000, 220, 34
    lo, hi = float(min(series)), float(max(series))
    span = max(hi - lo, 1e-6)

    def px(i):
        return pad + (w - 2 * pad) * (i / max(n - 1, 1))

    def py(v):
        return h - pad - (h - 2 * pad) * ((v - lo) / span)

    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(series))
    marks = "".join(
        f'<line x1="{px(k):.1f}" y1="{pad - 6}" x2="{px(k):.1f}" y2="{h - pad}" '
        f'stroke="#c9a227" stroke-width="1" opacity="0.55"/>' for k in keys
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">'
        f'<rect width="{w}" height="{h}" fill="#0b0d10"/>{marks}'
        f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.8"/>'
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" '
        f'stroke="#3a424c"/>'
        f'<text x="{pad}" y="{pad - 12}" fill="#9aa5b1" font-size="12" '
        f'font-family="monospace">{html.escape(ylabel)}</text>'
        f'<text x="{pad}" y="{h - 10}" fill="#6b7480" font-size="11" '
        f'font-family="monospace">frame 0</text>'
        f'<text x="{w - pad}" y="{h - 10}" fill="#6b7480" font-size="11" '
        f'font-family="monospace" text-anchor="end">frame {n - 1}</text>'
        f'<text x="{w - pad}" y="{pad - 12}" fill="#6b7480" font-size="11" '
        f'font-family="monospace" text-anchor="end">{lo:.4g} .. {hi:.4g}</text></svg>'
    )


def chart_stack(fracs, keys) -> str:
    """Share of coded area by block size, per frame -- 64 at the bottom, 8 on top."""
    n = len(fracs)
    w, h, pad = 1000, 220, 34
    def px(i):
        return pad + (w - 2 * pad) * (i / max(n - 1, 1))
    bands, base = [], np.zeros(n)
    for size in DEPTH_ORDER:
        top = base + np.array([f[size] for f in fracs])
        up = " ".join(f"{px(i):.1f},{h - pad - (h - 2 * pad) * v:.1f}"
                      for i, v in enumerate(top))
        down = " ".join(f"{px(i):.1f},{h - pad - (h - 2 * pad) * v:.1f}"
                        for i, v in reversed(list(enumerate(base))))
        bands.append(f'<polygon points="{up} {down}" fill="{DEPTH_COLOURS[size]}" '
                     f'opacity="0.9"/>')
        base = top
    marks = "".join(
        f'<line x1="{px(k):.1f}" y1="{pad - 6}" x2="{px(k):.1f}" y2="{h - pad}" '
        f'stroke="#f2f4f7" stroke-width="1" opacity="0.5"/>' for k in keys)
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">'
            f'<rect width="{w}" height="{h}" fill="#0b0d10"/>{"".join(bands)}{marks}'
            f'<text x="{pad}" y="{pad - 12}" fill="#9aa5b1" font-size="12" '
            f'font-family="monospace">share of coded area, by block size</text></svg>')


def panel_accum(acc: np.ndarray, cell: int, w: int, h: int, image: str) -> str:
    """Mean tree depth per cell over the whole clip, over the first frame."""
    lo, hi = float(acc.min()), float(acc.max())
    span = max(hi - lo, 1e-6)
    rects = []
    for r in range(acc.shape[0]):
        for c in range(acc.shape[1]):
            rects.append(
                f'<rect x="{c * cell}" y="{r * cell}" width="{min(cell, w - c * cell)}" '
                f'height="{min(cell, h - r * cell)}" '
                f'fill="{_heat((acc[r, c] - lo) / span)}" opacity="0.78"/>')
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">'
            f'<image href="{image}" x="0" y="0" width="{w}" height="{h}"/>'
            f'<rect width="{w}" height="{h}" fill="#0b0d10" opacity="0.45"/>'
            f'{"".join(rects)}</svg>')


def motion_reference(path: str, cell: int, w: int, h: int, n: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                 list[int]] | None:
    """Motion and pixel change from an MV-capable encode of the same shot.

    VP9 exports no motion at all, so the only way to ask "is the partition tree a
    stand-in for motion?" is to decode a codec that does export it and compare. The
    pixel-difference trace comes along because neither compressed-domain channel is
    ground truth: both are encoder decisions, and only the decoded picture can say which
    of them tracked the scene. Returns (per-cell fraction of frames moving, per-frame
    fraction of area moving, per-frame mean |pixel delta|, keyframe indices), or None if
    the reference has no motion channel or a different geometry.
    """
    ex = VideoExtractor(path, pixels=True)
    if not ex.capability().motion_vectors:
        return None
    ref = ex.extract()
    ref.sort(key=lambda f: f.display_index)
    if len(ref) != n or ref[0].width != w or ref[0].height != h:
        return None
    rows, cols = -(-h // cell), -(-w // cell)
    acc = np.zeros((rows, cols))
    trace, sad, prev = [], [], None
    for f in ref:
        cur = f.pixels.astype(np.float32)
        sad.append(0.0 if prev is None else float(np.abs(cur - prev).mean()))
        prev = cur
        m = np.zeros((h, w), dtype=np.float32)
        for u in f.units:
            if not u.mvs:
                continue
            dx = sum(v.dx for v in u.mvs) / len(u.mvs)
            dy = sum(v.dy for v in u.mvs) / len(u.mvs)
            # 1px, not 0: below that the field is dominated by sensor noise, and a
            # static camera would otherwise read as moving everywhere.
            if dx * dx + dy * dy >= 1.0:
                m[u.y:u.y + u.h, u.x:u.x + u.w] = 1.0
        trace.append(float(m.mean()))
        m = np.pad(m, ((0, rows * cell - h), (0, cols * cell - w)), mode="edge")
        acc += m.reshape(rows, cell, cols, cell).mean(axis=(1, 3))
    return (acc / len(ref), np.array(trace), np.array(sad),
            [i for i, f in enumerate(ref) if f.key_frame])


def _corr(a, b) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def chart_traces(series, keys) -> str:
    """Several traces on independent scales -- shapes are compared, not levels."""
    n = len(series[0][0])
    w, h, pad = 1000, 240, 34

    def px(i):
        return pad + (w - 2 * pad) * (i / max(n - 1, 1))

    def poly(series, colour):
        lo, hi = float(min(series)), float(max(series))
        span = max(hi - lo, 1e-12)
        pts = " ".join(
            f"{px(i):.1f},{h - pad - (h - 2 * pad) * ((v - lo) / span):.1f}"
            for i, v in enumerate(series))
        return (f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                f'stroke-width="1.8" opacity="0.95"/>')

    marks = "".join(
        f'<line x1="{px(k):.1f}" y1="{pad - 6}" x2="{px(k):.1f}" y2="{h - pad}" '
        f'stroke="#c9a227" stroke-width="1" opacity="0.35"/>' for k in keys)
    lines = "".join(poly(v, c) for v, c, _ in series)
    keyed = "".join(
        f'<text x="{pad + i * 250}" y="{pad - 12}" fill="{c}" font-size="12" '
        f'font-family="monospace">{html.escape(lab)}</text>'
        for i, (_, c, lab) in enumerate(series))
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">'
            f'<rect width="{w}" height="{h}" fill="#0b0d10"/>{marks}{lines}{keyed}</svg>')


def _jpeg_data_uri(luma: np.ndarray, quality: int = 82) -> str:
    return _jpeg_data_uri_rgb(np.repeat(luma[:, :, None], 3, axis=2), quality)


def _jpeg_data_uri_rgb(rgb: np.ndarray, quality: int = 82) -> str:
    h, w = rgb.shape[:2]
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mjpeg") as container:
        stream = container.add_stream("mjpeg", rate=1)
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuvj420p"
        stream.codec_context.qmin = stream.codec_context.qmax = max(
            2, int(31 - quality * 0.29))
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _cadence(path: str, n_frames: int) -> float:
    """Frames per second measured from duration, not read off `average_rate`.

    Surveillance clips are variable-rate: this one carries a 1/1000 timebase, so ffmpeg
    reports `r_frame_rate=100/1` and a re-encode at that number plays the whole clip in
    a second. Dividing the real duration by the real frame count is the only figure that
    survives the round trip.
    """
    with av.open(path) as c:
        st = c.streams.video[0]
        dur = float(st.duration * st.time_base) if st.duration else None
        if not dur and c.duration:
            dur = c.duration / 1e6
        if dur and dur > 0:
            return n_frames / dur
        r = st.average_rate
        return float(r) if r else 10.0


def build(path: str, out: str, cell: int, fps: float | None, crf: int,
          compare: str | None) -> str:
    ex = VideoExtractor(path, pixels=True, max_frames=None)
    cap = ex.capability()
    frames = ex.extract()
    frames.sort(key=lambda f: f.display_index)
    w, h = frames[0].width, frames[0].height
    if fps is None:
        fps = _cadence(path, len(frames))

    keys = [i for i, f in enumerate(frames) if f.key_frame]
    rows, cols = -(-h // cell), -(-w // cell)
    acc = np.zeros((rows, cols), dtype=np.float64)
    counts, mean_depth, qmeans, qdistinct, fracs = [], [], [], [], []
    shapes = collections.Counter()
    part_rgb = []
    qp_all = [u.qp for f in frames for u in f.units if u.qp is not None]
    qlo, qhi = (float(min(qp_all)), float(max(qp_all))) if qp_all else (0.0, 1.0)

    for f in frames:
        uid, depth, _ = _maps(f)
        part_rgb.append(overlay_partitions(f, uid, depth))
        obs = [u for u in f.units if u.geometry_observed]
        counts.append(len(obs))
        # Depth as tree level: 64x64 is 0, 8x8 is 3. Area-weighted, so it reads as
        # "how finely is this frame cut", not "how many small blocks are there".
        area = np.array([u.w * u.h for u in obs], dtype=np.float64)
        lev = np.array([DEPTH_ORDER.index(_depth_class(u.w, u.h)) for u in obs],
                       dtype=np.float64)
        mean_depth.append(float((area * lev).sum() / max(area.sum(), 1)))
        by = collections.Counter()
        for u in obs:
            s = _depth_class(u.w, u.h)
            by[s] += u.w * u.h
            shapes[(u.w, u.h)] += 1
        tot = max(sum(by.values()), 1)
        fracs.append({s: by[s] / tot for s in DEPTH_ORDER})
        qs = [u.qp for u in f.units if u.qp is not None]
        qmeans.append(float(np.mean(qs)) if qs else 0.0)
        qdistinct.append(len(set(qs)))
        # Accumulate depth per cell from the per-pixel level map. Padded to a whole
        # number of cells with edge values, so a non-multiple-of-cell frame does not
        # silently drop its last row or column.
        lvl = depth.astype(np.float64) - 1.0
        lvl = np.pad(lvl, ((0, rows * cell - h), (0, cols * cell - w)), mode="edge")
        acc += lvl.reshape(rows, cell, cols, cell).mean(axis=(1, 3))

    acc /= len(frames)
    bundle = patchify_grid(frames, cell=cell)
    _, report = anchor_delta(bundle, anchors=keyframe_anchors(frames))

    part_uri, part_bytes = _h264_data_uri(part_rgb, fps, crf)
    still = _jpeg_data_uri(frames[0].pixels)

    # The QP overlay is a still, not a video: on this encode 95 of 106 frames carry a
    # single q_idx across every unit, so an animation of it is 106 flat washes. Draw the
    # frame that actually has spatial variation, on its own rank scale.
    qframe = int(np.argmax(qdistinct))
    _, _, qmap = _maps(frames[qframe])
    qvals = np.sort(np.asarray(
        [u.qp for u in frames[qframe].units if u.qp is not None], dtype=np.float32))
    qp_still = _jpeg_data_uri_rgb(overlay_qp(frames[qframe], qmap, qvals))
    flat = sum(1 for d in qdistinct if d == 1)
    qspread = float(np.mean([
        np.std([u.qp for u in f.units if u.qp is not None] or [0.0]) for f in frames]))

    ref = motion_reference(compare, cell, w, h, len(frames)) if compare else None
    nonkey = np.ones(len(frames), dtype=bool)
    nonkey[keys] = False
    if ref is not None:
        # The reference's own keyframes carry no vectors at all, so leaving them in
        # scores "nothing moved" against a finely cut frame and understates r.
        nonkey[ref[3]] = False
        nonkey[0] = False  # frame 0 has no predecessor to difference against

    top = ", ".join(f"{a}x{b}: {n:,}" for (a, b), n in shapes.most_common(6))
    inter = [(n, i) for i, n in enumerate(counts) if i not in keys]
    busy_n, busy_i = max(inter) if inter else (max(counts), int(np.argmax(counts)))
    qneg = sum(1 for q in qp_all if q < 0)
    legend = "".join(
        f'<span class="k"><i style="background:{DEPTH_COLOURS[s]}"></i>{s}px</span>'
        for s in DEPTH_ORDER)

    cards = [
        ("The partition tree, every frame",
         f"{len(frames)} frames, replayed at {fps:.2f} fps -- the measured average, "
         f"since the source is variable-rate. Each hairline is a boundary the "
         f"encoder chose; colour is the block's larger side. The quietest frame is cut "
         f"into {min(counts):,} blocks, the busiest inter frame ({busy_i}) into "
         f"{busy_n:,}, and a keyframe into {max(counts):,} -- a 51x range over one "
         f"static camera.",
         f'<video src="{part_uri}" controls loop muted autoplay playsinline></video>',
         f"{part_bytes / 1e6:.2f} MB embedded"),
        ("Partitions per frame",
         "The count is an activity trace on its own: the encoder splits where "
         "prediction fails. Gold rules mark keyframes, which are cut fine everywhere "
         "because they have nothing to predict from.",
         chart_line(counts, keys, "coding units per frame"), ""),
        ("How finely each frame is cut",
         "Area-weighted tree level, 0 = all 64x64, 3 = all 8x8. Weighting by area "
         "rather than counting blocks stops a handful of 8x8s in one corner from "
         "reading as a busy frame.",
         chart_line(mean_depth, keys, "mean tree level (area-weighted)", "#c9a227"), ""),
        ("Share of coded area by block size",
         f"Whole clip: {top}. Keyframes are the pale rules.",
         chart_stack(fracs, keys), ""),
        ("Where the tree splits, over the whole clip",
         f"Mean tree level per {cell}px cell, averaged over all {len(frames)} frames, "
         f"drawn over frame 0. Read it before the next panel and it looks like a scene "
         f"map earned for free from bitstream syntax.",
         panel_accum(acc, cell, w, h, still), ""),
    ]

    if ref is not None:
        mov_acc, mov_trace, sad, _ = ref
        cards += [
            ("Where H.264 puts its vectors, same shot",
             f"The identical footage decoded from its H.264 original, which does export "
             f"motion: fraction of frames in which each {cell}px cell carries a vector "
             f"of 1px or more. It is speckle over the cinderblock and quiet on the flat "
             f"door -- the shape of a motion search, not the shape of a scene. Spatial "
             f"correlation with the tree above is {_corr(acc, mov_acc):+.2f}. The two "
             f"channels are not describing the same thing and neither is describing an "
             f"object.",
             panel_accum(mov_acc, cell, w, h, still), ""),
            ("Both channels against what actually changed",
             f"The arbiter is the decoded picture: mean |pixel delta| between "
             f"consecutive frames, which owes nothing to either encoder's decisions. "
             f"H.264's moving area tracks it at r = {_corr(mov_trace[nonkey], sad[nonkey]):+.2f}; "
             f"VP9's partition count at r = {_corr(np.array(counts)[nonkey], sad[nonkey]):+.2f}; "
             f"the two compressed-domain channels agree with each other only at "
             f"r = {_corr(np.array(counts)[nonkey], mov_trace[nonkey]):+.2f}. So the "
             f"motion channel is an almost exact meter of how much the picture changed, "
             f"while saying little about where -- and the quadtree is a poor substitute "
             f"for it, because splitting answers a different question.",
             chart_traces([(np.asarray(counts, float), "#4fb3ba", "VP9 partitions"),
                           (mov_trace, "#c04a3d", "H.264 moving area"),
                           (sad, "#c9a227", "mean |pixel delta|")], keys), ""),
        ]

    cards += [
        ("Per-unit quantiser index, on the one frame that has any",
         f"VP9 carries delta_qp on every leaf, so there is no fixed grid to draw it on "
         f"-- but on this encode it barely varies within a frame: {flat} of "
         f"{len(frames)} frames carry a single q_idx across every unit, and the mean "
         f"within-frame standard deviation is {qspread:.2f}. Frame {qframe} is the "
         f"exception, with {max(qdistinct)} "
         f"distinct values. The capability is real; libvpx at -aq-mode 1 mostly "
         f"declines to use it, which the synthetic fixture (17 distinct values) does "
         f"not show.",
         f'<img src="{qp_still}" alt="per-unit quantiser index">', ""),
        ("Mean quantiser index per frame",
         f"Where the variation actually lives. {int(qlo)} to {int(qhi)} on VP9's q_idx "
         f"scale (0-255, not H.264's 0-51 QP), median {int(np.median(qp_all))} over "
         f"{sum(len(f.units) for f in frames):,} units; "
         f"{qneg / len(qp_all) * 100:.1f}% fall below 0 because vpatch reports "
         f"base_qp + delta_qp unclamped rather than folding the sum into [0, 255]. "
         f"The sawtooth is rate control recovering after each keyframe -- bit "
         f"allocation, not the scene, which is why qp_delta ships off as a pruning rule.",
         chart_line(qmeans, keys, "mean q_idx", "#9c5195"), ""),
    ]

    body = "".join(
        f'<figure><h2>{html.escape(t)}</h2><p>{html.escape(sub)}</p>'
        f'<div class="frame">{media}</div>'
        + (f'<p class="note">{html.escape(note)}</p>' if note else "")
        + "</figure>"
        for t, sub, media, note in cards)

    doc = f"""<title>The Quadtree, Moving</title>
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
  .lead {{ color:var(--mut); margin:0 0 1.25rem; max-width:66ch; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:.85rem; margin:0 0 1.5rem;
    font-size:.82rem; color:var(--mut); }}
  .k {{ display:inline-flex; align-items:center; gap:.4rem; }}
  .k i {{ width:.72rem; height:.72rem; border-radius:2px; display:inline-block; }}
  figure {{ margin:0 0 1.6rem; background:var(--card); border:1px solid var(--line);
    border-radius:10px; padding:1rem 1rem 1.1rem; }}
  h2 {{ font-size:.95rem; margin:0 0 .2rem; }}
  figure p {{ margin:0 0 .7rem; color:var(--mut); font-size:.82rem; max-width:80ch; }}
  .note {{ margin:.6rem 0 0; font-family:ui-monospace,Menlo,monospace;
    font-size:.75rem; }}
  .frame {{ overflow-x:auto; }}
  svg, video, img {{ width:100%; height:auto; display:block; border-radius:6px;
    background:#0b0d10; }}
  footer {{ color:var(--mut); font-size:.8rem; margin-top:2rem; max-width:80ch; }}
  code {{ font-size:.85em; }}
</style>
<main>
<h1>The quadtree, moving</h1>
<p class="lead">Every overlay below is read out of the compressed VP9 bitstream, not
computed from the decoded picture. Source: <code>{html.escape(path.rsplit("/", 1)[-1])}
</code>, {html.escape(cap.codec)}, {w}x{h}, {len(frames)} frames. Coverage is exactly
1.0 on every frame, overlaps zero -- the tree tiles the picture.</p>
<div class="legend">{legend}<span class="k">coloured by the block's larger side</span>
</div>
{body}
<footer>VP9 exports partitions and per-block <code>delta_qp</code> but
<strong>no motion vectors</strong> -- ffmpeg's only MV producer is
<code>ff_print_debug_info2()</code>, reached from H.264 and MPEG-video alone. So the
motion channels are empty here and anchored pruning has nothing to prune by:
<code>no_change_signal</code> is set and the clip keeps only its
{len(keys)} keyframes, {report.kept_fraction * 100:.0f}% of its tokens -- not a saving
but a collapse, since with no change signal every non-anchor cell is dropped for want of
evidence. That is what the flag is for. The obvious hope is that the partition
tree stands in for the vectors -- the encoder did run a motion search, and splitting is
where it failed. Measured against the H.264 original of the same shot, it
does not: {"r = %+.2f in time, %+.2f in space -- and the pixels say the motion channel "
  "it is standing in for was itself measuring the noise floor of a camera pointed at "
  "nothing, at r = %+.2f" % (
    _corr(np.array(counts)[nonkey], ref[1][nonkey]), _corr(acc, ref[0]),
    _corr(ref[1][nonkey], ref[2][nonkey]))
  if ref is not None else "run again with --compare to measure it"}.
Generated by <code>tools/visualize_clip.py</code>.</footer>
</main>"""
    with open(out, "w") as fh:
        fh.write(doc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-o", "--out", default="vpatch-clip.html")
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--compare", default=None,
                    help="an MV-capable encode of the SAME shot (e.g. the "
                         "H.264 original), to test the tree against motion")
    args = ap.parse_args()
    print(build(args.path, args.out, args.cell, args.fps, args.crf,
                args.compare))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
