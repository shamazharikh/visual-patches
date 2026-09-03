"""Turn coding units into tokens on a fixed grid.

`grid` is the default and the conservative bet: one token per macroblock cell per frame,
carrying what the codec decided about that cell. The alternative (`native`, one token per
coding unit) is deferred -- measured against a dense 16x16 grid it is 0.90-1.19x the token
count, so it buys variable geometry rather than fewer tokens.

**The aggregation rule is the substance of this module.** A cell can contain up to four
8x8 partitions with different motion, and a plain mean over them destroys exactly the
signal worth keeping: partitions of (+4,0), (-4,0), (+4,0), (-4,0) average to (0,0), which
is byte-identical to a static cell. Every motion channel therefore carries an
area-weighted **standard deviation** alongside its mean. Mean says where the cell moved;
std says whether the codec found one motion or several, which is what "the encoder split
this macroblock" actually means.

Weights are overlap **areas**, so a 64x64 VP9 block contributes to sixteen cells in
proportion to how much of each it covers, and a unit is never counted as though it were a
whole cell.
"""

from __future__ import annotations

import math

import numpy as np

from vpatch.partition import MACROBLOCK, grid_shape
from vpatch.types import CodedFrame, Modality, PatchBundle, UnitKind

# Per-cell aggregate class. Distinct from UnitKind: a cell is a mixture, and codecs that
# export geometry with no intra/inter signal at all (VP9) must not be forced into a label.
KIND_UNKNOWN = -1
KIND_INTRA = 0
KIND_INTER = 1

FEATURE_LAYOUT: tuple[str, ...] = (
    # L0 motion, luma pixels. Raw: no reference index is exported, so there is no
    # temporal distance to normalise by, and dividing by an estimated one would be worse
    # than leaving it to a consumer that knows the GOP structure.
    "l0_dx_mean", "l0_dy_mean",
    "l0_dx_std", "l0_dy_std",     # variance-preserving: >0 means the codec split this cell
    "l0_mag_mean",
    # L1. A prediction-LIST index, not a time direction -- in low-delay B and B-pyramid
    # both lists can point into the past, so these are kept as their own channels rather
    # than folded into a signed "forward/backward" pair.
    "l1_dx_mean", "l1_dy_mean",
    "l1_dx_std", "l1_dy_std",
    "l1_mag_mean",
    # Validity fractions. Every mean above is 0 where its fraction is 0; these channels
    # are what distinguish "measured zero motion" from "no measurement".
    "l0_frac", "l1_frac", "bipred_frac",
    "inter_frac",       # area with at least one MV record
    "observed_frac",    # area from real partitions, not grid fill
    # Partition geometry: how finely the encoder split this cell.
    "log2_area_mean", "log2_area_min", "unit_count",
    # Coding cost. `qp_std` is structurally zero at cell=16 for H.264, because ffmpeg
    # reports QP on exactly the macroblock grid, so every unit in a cell shares one value.
    # It carries signal only at larger cells -- measured max 0.0 at cell=16, 13.0 at 32.
    "qp_mean", "qp_std", "qp_frac",
)
D = len(FEATURE_LAYOUT)
_IDX = {name: i for i, name in enumerate(FEATURE_LAYOUT)}


def _unit_arrays(units, cell, width, height):
    """Flatten units into parallel arrays, plus per-unit per-list motion moments.

    Motion is reduced to sums here rather than kept per-vector because every MV on a
    unit shares that unit's overlap weight, so `w += a*n`, `s += a*sum(v)`,
    `ss += a*sum(v^2)` is exactly what a per-vector loop would accumulate.
    """
    n = len(units)
    ux = np.empty(n, dtype=np.int64)
    uy = np.empty(n, dtype=np.int64)
    uw = np.empty(n, dtype=np.int64)
    uh = np.empty(n, dtype=np.int64)
    log2a = np.empty(n, dtype=np.float64)
    observed = np.empty(n, dtype=np.float64)
    inter = np.empty(n, dtype=np.float64)
    unknown = np.empty(n, dtype=np.float64)
    bipred = np.empty(n, dtype=np.float64)
    qp_has = np.zeros(n, dtype=np.float64)
    qp_s = np.zeros(n, dtype=np.float64)
    qp_ss = np.zeros(n, dtype=np.float64)
    # [list][0..3] = count, sum dx, sum dy, sum mag  and their squares
    mv = np.zeros((2, 7, n), dtype=np.float64)

    for i, u in enumerate(units):
        ux[i], uy[i], uw[i], uh[i] = u.x, u.y, u.w, u.h
        area = u.w * u.h
        log2a[i] = math.log2(area) if area > 0 else 0.0
        observed[i] = 1.0 if u.geometry_observed else 0.0
        inter[i] = 1.0 if u.kind is UnitKind.INTER else 0.0
        unknown[i] = 1.0 if u.kind is None else 0.0
        if u.qp is not None:
            qp_has[i] = 1.0
            qp_s[i] = float(u.qp)
            qp_ss[i] = float(u.qp) ** 2
        lists = set()
        for m in u.mvs:
            li = 1 if m.list_idx else 0
            lists.add(li)
            mag = math.hypot(m.dx, m.dy)
            mv[li, 0, i] += 1.0
            mv[li, 1, i] += m.dx
            mv[li, 2, i] += m.dy
            mv[li, 3, i] += mag
            mv[li, 4, i] += m.dx * m.dx
            mv[li, 5, i] += m.dy * m.dy
            mv[li, 6, i] += mag * mag
        bipred[i] = 1.0 if len(lists) > 1 else 0.0

    return {"x": ux, "y": uy, "w": uw, "h": uh, "log2a": log2a, "observed": observed,
            "inter": inter, "unknown": unknown, "bipred": bipred, "qp_has": qp_has,
            "qp_s": qp_s, "qp_ss": qp_ss, "mv": mv}


def _pairs(a, cell, width, height, cols):
    """Expand units into (unit, cell) pairs with their overlap areas.

    A unit touches one cell in the common case and up to sixteen for a VP9 64x64
    superblock over a 16px grid, so the pair list is built by repeat/arithmetic rather
    than a nested Python loop -- at 1080p there are ~8k units per frame and the loop
    version cost 4x the decode it was reading from.
    """
    c0x, c1x = a["x"] // cell, (a["x"] + a["w"] - 1) // cell
    c0y, c1y = a["y"] // cell, (a["y"] + a["h"] - 1) // cell
    nx, ny = c1x - c0x + 1, c1y - c0y + 1
    npairs = nx * ny
    total = int(npairs.sum())
    if total == 0:
        return None

    idx = np.repeat(np.arange(len(npairs)), npairs)
    starts = np.cumsum(npairs) - npairs
    k = np.arange(total) - np.repeat(starts, npairs)
    nx_r = np.repeat(nx, npairs)
    cx = np.repeat(c0x, npairs) + k % nx_r
    cy = np.repeat(c0y, npairs) + k // nx_r

    x, y = a["x"][idx], a["y"][idx]
    ow = np.minimum(x + a["w"][idx], np.minimum((cx + 1) * cell, width)) - np.maximum(x, cx * cell)
    oh = np.minimum(y + a["h"][idx], np.minimum((cy + 1) * cell, height)) - np.maximum(y, cy * cell)
    area = (np.maximum(ow, 0) * np.maximum(oh, 0)).astype(np.float64)

    keep = area > 0
    return idx[keep], (cy * cols + cx)[keep], area[keep]


def frame_features(frame: CodedFrame, cell: int = MACROBLOCK
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate one frame's units onto the grid.

    Returns (features [R, C, D], kinds [R, C] int8, coords [R, C, 4] float32).
    """
    rows, cols = grid_shape(frame.width, frame.height, cell)
    ncell = rows * cols
    feats = np.zeros((ncell, D), dtype=np.float64)
    area_tot = np.zeros(ncell, dtype=np.float64)
    inter_a = np.zeros(ncell, dtype=np.float64)
    unknown_a = np.zeros(ncell, dtype=np.float64)
    log2_min = np.full(ncell, np.inf, dtype=np.float64)

    a = _unit_arrays(frame.units, cell, frame.width, frame.height) if frame.units else None
    expanded = _pairs(a, cell, frame.width, frame.height, cols) if a is not None else None

    if expanded is not None:
        idx, flat, area = expanded

        def acc(vals: np.ndarray) -> np.ndarray:
            return np.bincount(flat, weights=area * vals[idx], minlength=ncell)

        area_tot = np.bincount(flat, weights=area, minlength=ncell)
        denom = np.maximum(area_tot, 1e-12)
        inter_a = acc(a["inter"])
        unknown_a = acc(a["unknown"])

        np.minimum.at(log2_min, flat, a["log2a"][idx])

        for li, tag in ((0, "l0"), (1, "l1")):
            m = a["mv"][li]
            w = acc(m[0])
            wsafe = np.maximum(w, 1e-12)
            for j, (mean_name, std_name) in enumerate(
                    ((f"{tag}_dx_mean", f"{tag}_dx_std"),
                     (f"{tag}_dy_mean", f"{tag}_dy_std"),
                     (f"{tag}_mag_mean", None))):
                mean = acc(m[1 + j]) / wsafe
                mean = np.where(w == 0, 0.0, mean)
                feats[:, _IDX[mean_name]] = mean
                if std_name is not None:
                    # Clamped: E[x^2] - E[x]^2 goes slightly negative under float
                    # cancellation when every sample in a cell is identical, which is
                    # the common case.
                    var = np.maximum(acc(m[4 + j]) / wsafe - mean * mean, 0.0)
                    feats[:, _IDX[std_name]] = np.where(w == 0, 0.0, np.sqrt(var))
            feats[:, _IDX[f"{tag}_frac"]] = np.minimum(w / denom, 1.0)

        feats[:, _IDX["bipred_frac"]] = np.minimum(acc(a["bipred"]) / denom, 1.0)
        feats[:, _IDX["inter_frac"]] = np.minimum(inter_a / denom, 1.0)
        feats[:, _IDX["observed_frac"]] = np.minimum(acc(a["observed"]) / denom, 1.0)
        feats[:, _IDX["log2_area_mean"]] = acc(a["log2a"]) / denom
        feats[:, _IDX["unit_count"]] = np.bincount(flat, minlength=ncell)

        qw = acc(a["qp_has"])
        qsafe = np.maximum(qw, 1e-12)
        qmean = np.where(qw == 0, 0.0, acc(a["qp_s"]) / qsafe)
        qvar = np.maximum(acc(a["qp_ss"]) / qsafe - qmean * qmean, 0.0)
        feats[:, _IDX["qp_mean"]] = qmean
        feats[:, _IDX["qp_std"]] = np.where(qw == 0, 0.0, np.sqrt(qvar))
        feats[:, _IDX["qp_frac"]] = np.minimum(qw / denom, 1.0)

    feats[:, _IDX["log2_area_min"]] = np.where(np.isinf(log2_min), 0.0, log2_min)

    # A cell is INTER if any of its area carried motion; INTRA if it carried none and the
    # codec was capable of saying so; UNKNOWN when the geometry came from a codec that
    # exports no intra/inter signal, or when the cell has no units at all.
    kinds = np.where(inter_a > 0, KIND_INTER,
                     np.where(unknown_a > 0, KIND_UNKNOWN,
                              np.where(area_tot > 0, KIND_INTRA, KIND_UNKNOWN))
                     ).astype(np.int8)

    # Coordinates are normalised to the VISIBLE frame, after clipping. The last row and
    # column are partial where the frame is not a multiple of the cell, so their centres
    # and sizes come from the clipped extent rather than being assumed square.
    xs = np.arange(cols) * cell
    ys = np.arange(rows) * cell
    x1 = np.minimum(xs + cell, frame.width)
    y1 = np.minimum(ys + cell, frame.height)
    cxs = (xs + x1) / 2 / frame.width
    cys = (ys + y1) / 2 / frame.height
    ws = (x1 - xs) / frame.width
    hs = (y1 - ys) / frame.height
    coords = np.empty((rows, cols, 4), dtype=np.float32)
    coords[..., 0] = cxs[None, :]
    coords[..., 1] = cys[:, None]
    coords[..., 2] = ws[None, :]
    coords[..., 3] = hs[:, None]

    return (feats.reshape(rows, cols, D).astype(np.float32),
            kinds.reshape(rows, cols), coords)


def patchify_grid(frames: list[CodedFrame], *, cell: int = MACROBLOCK,
                  meta: dict | None = None) -> PatchBundle:
    """One token per grid cell per frame, in display order."""
    ordered = sorted(frames, key=lambda f: f.display_index)
    feats_all, kinds_all, coords_all, times_all = [], [], [], []
    for t, frame in enumerate(ordered):
        f, k, c = frame_features(frame, cell)
        n = f.shape[0] * f.shape[1]
        feats_all.append(f.reshape(n, D))
        kinds_all.append(k.reshape(n))
        coords_all.append(c.reshape(n, 4))
        times_all.append(np.full(n, t, dtype=np.int32))

    if not feats_all:
        raise ValueError("no frames to patchify")

    features = np.concatenate(feats_all).astype(np.float32)
    return PatchBundle(
        features=features,
        coords=np.concatenate(coords_all).astype(np.float32),
        times=np.concatenate(times_all).astype(np.int32),
        kinds=np.concatenate(kinds_all).astype(np.int8),
        # One sample = one clip. Packing several clips concatenates bundles and their
        # seq_lens; the boundary is not recoverable from the features alone, which is why
        # it is carried rather than derived.
        seq_lens=[len(features)],
        modality=Modality.VIDEO.value,
        meta={
            "strategy": "grid",
            "cell": cell,
            "feature_layout": list(FEATURE_LAYOUT),
            "mv_units": "luma_pixels",
            "ref_identity_available": False,
            "n_frames": len(ordered),
            "grid": grid_shape(ordered[0].width, ordered[0].height, cell),
            # Needed to recover a cell index from normalised coords: the last row and
            # column are partial whenever `cell` does not divide the frame, so scaling a
            # centre by the grid size does not invert.
            "frame_size": (ordered[0].width, ordered[0].height),
            **(meta or {}),
        },
    )
