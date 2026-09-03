# vpatch

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![PyAV](https://img.shields.io/badge/av-%E2%89%A518.1-blue)](https://github.com/PyAV-Org/PyAV)
[![ffmpeg](https://img.shields.io/badge/ffmpeg-public%20API%20only-brightgreen)](#what-each-codec-actually-gives-you)
[![scope](https://img.shields.io/badge/scope-extraction%20only%20%E2%80%94%20no%20model%2C%20no%20weights-lightgrey)](#)

Codecs already segment every frame and estimate its motion. Decoding to pixels and cutting
a uniform patch grid throws that work away. `vpatch` reads it back out of the compressed
bitstream — motion vectors, per-block QP, partition geometry, frame structure — and
exposes it as features aligned to a patch grid, so a VLM pipeline can use motion and
coding complexity as a cheap prior instead of re-deriving them from pixels.

Extraction library only. No model, no training, no weights. Public ffmpeg API only —
no patched libavcodec, no hand-rolled bitstream parser.

```bash
uv run vpatch probe clip.mp4     # what this file's codec actually exports
uv run vpatch stats clip.mp4     # tokens/second against a real VLM baseline
```

```python
from vpatch.backends.ffmpeg_video import extract_video
from vpatch.patchify import patchify_grid
from vpatch.sampling import anchor_delta, keyframe_anchors

frames = extract_video("clip.mp4", pixels=False)
bundle = patchify_grid(frames, cell=28)
pruned, report = anchor_delta(bundle, anchors=keyframe_anchors(frames))
print(report.kept_fraction, report.cells_never_kept)
```

## What each codec actually gives you

Measured on this machine against av 18.1.0 / libavcodec 62.28.102, not read off a spec.
The three axes are independent — no codec has all of them.

| Codec | Motion vectors | Partitions | Per-block QP |
|---|---|---|---|
| **H.264** | **yes** | `{16x16, 16x8, 8x16, 8x8}` only, two depths | **yes**, on the macroblock grid |
| **VP9** | no | **yes** — 64/32/16/8 quadtree, exact tiling | **yes**, per-block `delta_qp` |
| HEVC / AV1 / VP8 | no | no | no |

HEVC exports **nothing**, at every ffmpeg version. The sole producer of motion-vector side
data is `ff_print_debug_info2()` in `libavcodec/mpegutils.c`, and its only call sites are
`h264dec.c` and `mpegvideo_dec.c`. Those codecs still yield frame types, ordering, and a
uniform grid — and say so, rather than emitting zeros that look like measurements.

VP9's tree is only exported when the bitstream uses segmentation (`-aq-mode 1`); a
default-CRF encode yields zero blocks. `probe --probe-frames N` trial-decodes to report
what a file *does*, separately from what its codec *could*.

## Measured token cost

`grid` is a feature substitution, not a compression scheme: tokens per frame are
`ceil(W/cell) * ceil(H/cell)`, a function of resolution and cell size alone. Against
Qwen2-VL's real defaults (14px patches, 2x2 merge, 2fps, temporal patch 2) the ratio at
equal temporal sampling is exactly `(28/cell)^2`.

Reduction comes from anchored pruning, and it is content-dependent:

| content | vs baseline |
|---|---|
| near-static scene, one moving object | **0.13x** |
| 1080p synthetic | 0.41x |
| **70 real surveillance clips, median** | **0.22x** (p10 0.13x, p90 0.42x) |
| full-frame pan | 0.94x |

Worst case is roughly parity. Surveillance footage sits at the good end because most of
the frame does not change.

## Design rules

The whole library exists to avoid one failure: a plausible-looking number where the honest
answer is *no data*.

- **Coverage is a rasterised occupancy mask, never a sum of areas.** ffmpeg emits one MV
  record per (block, prediction list), so raw record areas sum to 1.70x the frame on a
  B-frame.
- **`dst_x`/`dst_y` is the block centre.** After conversion, 612/612 blocks land on their
  own grid; before it, 0/612.
- **`source` is a prediction-list index, not a time direction.** 7.7% of blocks are
  L1-only, and labelling those "from the future" is wrong 1 block in 13.
- **No reference identity is exported**, so motion cannot be normalised by temporal
  distance. Vectors are emitted raw, in luma pixels.
- **Absence is flagged, never zero-filled.** `geometry_observed`, `l0_frac`, `qp_frac`,
  `DropReport.no_change_signal` all separate "measured, and it was zero" from "never
  measured".
- **Pruning is anchored.** Every cell is kept on keyframes, so a missing token means
  "unchanged since the last anchor" rather than "unknown". Dropping every low-motion cell
  instead deletes 27 of 50 frames on a near-static clip.
- **Extraction is a pure function of (file bytes, kwargs)**, byte-identical across
  processes and thread counts.

## Looking at it

Two renderers, both writing one self-contained HTML file with everything embedded:

```bash
uv run python tools/visualize.py CLIP.mp4 -o panels.html          # one frame, every channel
uv run python tools/visualize_clip.py CLIP.webm -o clip.html \
    --compare ORIGINAL.mp4                                        # every frame, one channel
```

`visualize.py` freezes a frame and draws partitions, motion, QP, the aggregated grid
tokens and the pruning mask over it. `visualize_clip.py` is the other axis — it re-encodes
the partition overlay as a video so a whole clip plays, and adds per-frame traces. It is
aimed at VP9, whose only contribution is a quadtree, and a quadtree says nothing until you
watch it move.

`--compare` takes an MV-capable encode of the *same shot* and turns the page into a
measurement. On real surveillance (a 720p stairwell camera, 106 frames, transcoded to VP9
at `-aq-mode 1`) it reports:

| | r |
|---|---|
| H.264 moving area vs. mean \|pixel delta\| | **+0.98** |
| VP9 partition count vs. mean \|pixel delta\| | +0.33 |
| VP9 partition count vs. H.264 moving area | +0.29 |
| VP9 split map vs. H.264 vector map, per cell | **−0.01** |

So the motion channel is an almost exact meter of *how much* a frame changed while
saying little about *where* — on that clip its vectors are speckle over textured wall,
on frames that are visually identical. The partition tree is a poor stand-in for it: it
traces structural edges, which is a different question. Two other things that clip
showed and the synthetic fixtures did not: libvpx at `-aq-mode 1` emits a **single**
`q_idx` for every unit on 95 of 106 frames (the fixture has 17 distinct values), and
VP9's rectangular partitions (`64x16`, `32x16`, …) do occur in the wild, at about 0.13%
of blocks.

## vLLM

Documentation, not a plugin — see [docs/vllm-integration.md](docs/vllm-integration.md).
`vpatch` never imports vLLM. The working seam is precomputed embeddings via
`--enable-mm-embeds`; the missing piece is a trained projection from ~21 codec channels
into a backbone's embedding space, which this project deliberately does not ship.

```bash
uv run --with vllm python tools/check_vllm_seam.py   # re-verify the doc's claims
```

## Tests

```bash
uv run pytest
```

Fixtures are committed binaries pinned by sha256 and regenerated only by
`tools/regen_fixtures.sh` at `-threads 1` — libx264 output depends on host thread count,
so generating them at test time would make every numeric assertion flake on a machine with
a different core count.
