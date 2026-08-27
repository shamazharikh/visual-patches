"""`vpatch probe` and `vpatch stats`.

`stats` exists to answer one question with numbers instead of a claim: is a
compressed-domain token stream cheaper than the pixel patches a VLM would otherwise cut?
The v1 plan asserted it was, on the theory that token count follows bitrate. Measured, it
does not -- so this command reports the comparison rather than assuming it, and the
baseline it compares against is a real VLM's default preprocessing, not a strawman.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import av
import numpy as np

from vpatch.backends.ffmpeg_video import (
    VideoExtractor,
    max_units_per_frame,
    open_container,
)
from vpatch.partition import MACROBLOCK, grid_shape
from vpatch.patchify import FEATURE_LAYOUT, patchify_grid
from vpatch.sampling import anchor_delta, budget, keyframe_anchors

# Qwen2-VL / Qwen2.5-VL video defaults: 14x14 ViT patches, 2x2 spatial merge (so 28 luma
# pixels per token side), 2 fps sampling, temporal patch size 2 (two sampled frames per
# token). This is the number to beat; a "one token per 16x16 of every frame" strawman
# would make any result look good.
BASELINE_PIXELS_PER_TOKEN_SIDE = 28
BASELINE_FPS = 2.0
BASELINE_TEMPORAL_MERGE = 2


def _probe_container(path: str) -> dict:
    with open_container(path) as container:
        st = container.streams.video[0]
        rate = st.average_rate or st.guessed_rate
        fps = float(rate) if rate else 0.0
        duration = None
        if st.duration is not None and st.time_base:
            duration = float(st.duration * st.time_base)
        elif container.duration:
            duration = container.duration / av.time_base
        return {
            "codec": st.codec_context.name,
            "width": st.codec_context.width,
            "height": st.codec_context.height,
            "fps": fps,
            "n_frames": st.frames or None,
            "duration_s": duration,
        }


def _baseline_tokens_per_second(width: int, height: int, *, fps: float = BASELINE_FPS,
                                side: int = BASELINE_PIXELS_PER_TOKEN_SIDE,
                                temporal_merge: int = BASELINE_TEMPORAL_MERGE) -> float:
    per_frame = math.ceil(width / side) * math.ceil(height / side)
    return per_frame * (fps / temporal_merge)


def cmd_probe(args: argparse.Namespace) -> int:
    info = _probe_container(args.path)
    ex = VideoExtractor(args.path, pixels=False, fill_holes=False,
                        max_frames=args.probe_frames or None)
    cap = ex.capability(probe_frames=args.probe_frames)
    out = {
        **info,
        "motion_vectors": cap.motion_vectors,
        "partitions": cap.partitions,
        "per_block_qp": cap.per_block_qp,
        "degraded": cap.degraded,
        # Declared capability is read off the codec name; `observed` is what this file
        # actually emitted. They differ: VP9 exports its tree only when the bitstream uses
        # segmentation, so a default-CRF VP9 probes partitions=True, observed=False.
        "observed_geometry": cap.observed,
        "max_units_per_frame": max_units_per_frame(info["width"], info["height"]),
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    info = _probe_container(args.path)
    ex = VideoExtractor(args.path, pixels=False, max_frames=args.max_frames)
    cap = ex.capability()
    frames = ex.extract()
    if not frames:
        print("no frames decoded", file=sys.stderr)
        return 1

    bundle = patchify_grid(frames, cell=args.cell)
    f = bundle.features
    idx = {n: i for i, n in enumerate(FEATURE_LAYOUT)}
    rows, cols = grid_shape(info["width"], info["height"], args.cell)
    per_frame = rows * cols
    n_frames = len(frames)

    fps = info["fps"] or 0.0
    duration = n_frames / fps if fps else None
    if duration is None or duration <= 0:
        print("cannot determine duration; tokens/second is undefined", file=sys.stderr)
        return 1

    baseline = _baseline_tokens_per_second(info["width"], info["height"], fps=args.baseline_fps)

    # Rate at which the baseline actually looks at frames, so the temporal comparison is
    # like for like: sampling vpatch at native frame rate against a 1-fps baseline would
    # be comparing token budgets AND frame rates at once.
    sampled_fps = args.baseline_fps / BASELINE_TEMPORAL_MERGE

    observed = f[:, idx["observed_frac"]] > 0

    def per_sec(mask_frac: float, rate: float) -> float:
        return per_frame * mask_frac * rate

    anchors = keyframe_anchors(frames)
    pruned, report = anchor_delta(bundle, anchors=anchors,
                                  motion_threshold=args.motion_threshold,
                                  qp_delta=args.qp_delta)
    capped = None
    if args.max_tokens is not None:
        capped, cap_report = budget(pruned, args.max_tokens, anchors=anchors)

    variants: list[tuple[str, float | None]] = [
        ("grid, every frame", per_sec(1.0, fps)),
        (f"grid, sampled @{sampled_fps:g}fps", per_sec(1.0, sampled_fps)),
        (f"observed only @{sampled_fps:g}fps", per_sec(observed.mean(), sampled_fps)),
        # None, not 0.0. On a codec with no motion export there is nothing to prune by,
        # and printing 0.00x there would read as "pruned everything".
        (f"anchor-delta @{sampled_fps:g}fps",
         per_sec(report.kept_fraction, sampled_fps) if cap.motion_vectors else None),
    ]
    if capped is not None:
        variants.append((f"anchor-delta + cap {args.max_tokens} @{sampled_fps:g}fps",
                         per_sec(cap_report.kept_fraction * report.kept_fraction,
                                 sampled_fps)))

    print(f"{args.path}")
    print(f"  {info['codec']} {info['width']}x{info['height']} "
          f"{fps:.3f}fps  {n_frames} frames  {duration:.2f}s")
    print(f"  grid {rows}x{cols} = {per_frame} cells/frame, D={f.shape[1]}")
    print(f"  coverage mean {np.mean([fr.coverage for fr in frames]):.4f}, "
          f"observed cells {observed.mean() * 100:.1f}%, "
          f"kept after delta pruning {report.kept_fraction * 100:.1f}%")
    print()
    print(f"  {'variant':<34} {'tok/s':>10} {'vs baseline':>12}")
    print(f"  {'-' * 34} {'-' * 10:>10} {'-' * 12:>12}")
    print(f"  {'BASELINE 28px patch @%gfps' % args.baseline_fps:<34} {baseline:10.1f} "
          f"{1.0:11.2f}x")
    for name, tps in variants:
        if tps is None:
            print(f"  {name:<34} {'n/a':>10} {'':>12}   ({cap.codec} exports no motion)")
        else:
            print(f"  {name:<34} {tps:10.1f} {tps / baseline:11.2f}x")
    print()
    print(f"  anchors: {len(anchors)} keyframe(s) of {n_frames} frames -- every cell is")
    print("  kept on those, so a missing token means 'unchanged since the last anchor'")
    print(f"  rather than 'unknown'. cells never emitted: {report.cells_never_kept}, "
          f"frames left empty: {report.frames_emptied}")
    for rule, count in sorted(report.by_rule.items(), key=lambda kv: -kv[1]):
        print(f"    {rule:<28} {count:>8}")
    if report.no_change_signal:
        print(f"  WARNING: {cap.codec} exported no motion anywhere in this clip, so no "
              "cell could")
        print("  ever count as changed and only anchor frames survived. The kept "
              "fraction above")
        print("  is an empty result, not a compression win.")
    if capped is not None and cap_report.budget_below_anchor_floor:
        print("  WARNING: token cap is below the anchor floor; anchors were dropped and")
        print("  the 'absence means unchanged' contract no longer holds.")

    if args.json:
        print(json.dumps({
            "path": args.path, **info, "n_frames_decoded": n_frames,
            "cell": args.cell, "cells_per_frame": per_frame, "D": int(f.shape[1]),
            "baseline_tokens_per_s": baseline,
            "motion_vectors_available": cap.motion_vectors,
            "drop_report": pruned.meta["drop_report"],
            "variants": {name: tps for name, tps in variants},
            "ratios": {name: (None if tps is None else tps / baseline)
                       for name, tps in variants},
        }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vpatch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="report what this file's codec actually exports")
    p.add_argument("path")
    p.add_argument("--probe-frames", type=int, default=8,
                   help="trial-decode this many frames to confirm declared capability")
    p.set_defaults(func=cmd_probe)

    s = sub.add_parser("stats", help="tokens per second of video against a VLM baseline")
    s.add_argument("path")
    s.add_argument("--cell", type=int, default=MACROBLOCK)
    s.add_argument("--max-frames", type=int, default=None)
    s.add_argument("--baseline-fps", type=float, default=BASELINE_FPS)
    s.add_argument("--motion-threshold", type=float, default=0.5,
                   help="luma pixels of mean L0 motion below which a cell counts as static")
    s.add_argument("--qp-delta", type=float, default=None,
                   help="also keep cells whose QP moved this far from their anchor. Off "
                        "by default: rate control wanders QP every frame even in a "
                        "static scene, so at qp_delta=2 it keeps 95%% of a near-static "
                        "clip that motion alone prunes to 12%%.")
    s.add_argument("--max-tokens", type=int, default=None,
                   help="hard token cap applied after delta pruning")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_stats)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
