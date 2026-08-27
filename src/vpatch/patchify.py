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


class _Acc:
    """Area-weighted first and second moments over one grid."""

    __slots__ = ("w", "s", "ss")

    def __init__(self, rows: int, cols: int, n: int):
        self.w = np.zeros((rows, cols), dtype=np.float64)
        self.s = np.zeros((rows, cols, n), dtype=np.float64)
        self.ss = np.zeros((rows, cols, n), dtype=np.float64)

    def add(self, r: int, c: int, area: float, vals: tuple[float, ...]) -> None:
        self.w[r, c] += area
        v = np.asarray(vals, dtype=np.float64)
        self.s[r, c] += area * v
        self.ss[r, c] += area * v * v

    def mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        w = np.maximum(self.w, 1e-12)[..., None]
        mean = self.s / w
        # Clamped: the identity E[x^2] - E[x]^2 goes slightly negative under float
        # cancellation when every sample in a cell is identical, which is the common case.
        var = np.maximum(self.ss / w - mean * mean, 0.0)
        empty = (self.w == 0.0)[..., None]
        return np.where(empty, 0.0, mean), np.where(empty, 0.0, np.sqrt(var))


def frame_features(frame: CodedFrame, cell: int = MACROBLOCK
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate one frame's units onto the grid.

    Returns (features [R, C, D], kinds [R, C] int8, coords [R, C, 4] float32).
    """
    rows, cols = grid_shape(frame.width, frame.height, cell)
    l0 = _Acc(rows, cols, 3)   # dx, dy, magnitude
    l1 = _Acc(rows, cols, 3)
    geom = _Acc(rows, cols, 1)  # log2 area
    qp = _Acc(rows, cols, 1)

    area_tot = np.zeros((rows, cols), dtype=np.float64)
    inter_a = np.zeros((rows, cols), dtype=np.float64)
    observed_a = np.zeros((rows, cols), dtype=np.float64)
    bipred_a = np.zeros((rows, cols), dtype=np.float64)
    unknown_a = np.zeros((rows, cols), dtype=np.float64)
    log2_min = np.full((rows, cols), np.inf, dtype=np.float64)
    count = np.zeros((rows, cols), dtype=np.float64)

    for u in frame.units:
        lists = {m.list_idx for m in u.mvs}
        is_bipred = len(lists) > 1
        log2_area = math.log2(u.area) if u.area > 0 else 0.0

        # A unit spans one cell in the common case (H.264 partitions are <= 16x16 and
        # aligned to their own size) and up to 16 for a VP9 64x64 superblock.
        for r in range(u.y // cell, (u.y + u.h - 1) // cell + 1):
            cy0, cy1 = r * cell, min((r + 1) * cell, frame.height)
            oh = min(u.y + u.h, cy1) - max(u.y, cy0)
            if oh <= 0:
                continue
            for c in range(u.x // cell, (u.x + u.w - 1) // cell + 1):
                cx0, cx1 = c * cell, min((c + 1) * cell, frame.width)
                ow = min(u.x + u.w, cx1) - max(u.x, cx0)
                if ow <= 0:
                    continue
                a = float(ow * oh)
                area_tot[r, c] += a
                count[r, c] += 1.0
                geom.add(r, c, a, (log2_area,))
                log2_min[r, c] = min(log2_min[r, c], log2_area)
                if u.geometry_observed:
                    observed_a[r, c] += a
                if u.kind is UnitKind.INTER:
                    inter_a[r, c] += a
                elif u.kind is None:
                    unknown_a[r, c] += a
                if is_bipred:
                    bipred_a[r, c] += a
                if u.qp is not None:
                    qp.add(r, c, a, (float(u.qp),))
                for m in u.mvs:
                    acc = l1 if m.list_idx else l0
                    acc.add(r, c, a, (m.dx, m.dy, math.hypot(m.dx, m.dy)))

    feats = np.zeros((rows, cols, D), dtype=np.float32)
    denom = np.maximum(area_tot, 1e-12)

    for acc, tag in ((l0, "l0"), (l1, "l1")):
        mean, std = acc.mean_std()
        feats[..., _IDX[f"{tag}_dx_mean"]] = mean[..., 0]
        feats[..., _IDX[f"{tag}_dy_mean"]] = mean[..., 1]
        feats[..., _IDX[f"{tag}_dx_std"]] = std[..., 0]
        feats[..., _IDX[f"{tag}_dy_std"]] = std[..., 1]
        feats[..., _IDX[f"{tag}_mag_mean"]] = mean[..., 2]
        feats[..., _IDX[f"{tag}_frac"]] = np.minimum(acc.w / denom, 1.0)

    feats[..., _IDX["bipred_frac"]] = np.minimum(bipred_a / denom, 1.0)
    feats[..., _IDX["inter_frac"]] = np.minimum(inter_a / denom, 1.0)
    feats[..., _IDX["observed_frac"]] = np.minimum(observed_a / denom, 1.0)

    gmean, _ = geom.mean_std()
    feats[..., _IDX["log2_area_mean"]] = gmean[..., 0]
    feats[..., _IDX["log2_area_min"]] = np.where(np.isinf(log2_min), 0.0, log2_min)
    feats[..., _IDX["unit_count"]] = count

    qmean, qstd = qp.mean_std()
    feats[..., _IDX["qp_mean"]] = qmean[..., 0]
    feats[..., _IDX["qp_std"]] = qstd[..., 0]
    feats[..., _IDX["qp_frac"]] = np.minimum(qp.w / denom, 1.0)

    # A cell is INTER if any of its area carried motion; INTRA if it carried none and the
    # codec was capable of saying so; UNKNOWN when the geometry came from a codec that
    # exports no intra/inter signal, or when the cell has no units at all.
    kinds = np.where(inter_a > 0, KIND_INTER,
                     np.where(unknown_a > 0, KIND_UNKNOWN,
                              np.where(area_tot > 0, KIND_INTRA, KIND_UNKNOWN))
                     ).astype(np.int8)

    # Coordinates are normalised to the VISIBLE frame, after clipping. The last row and
    # column are partial where the frame is not a multiple of the cell, so their centres
    # and sizes are computed from the clipped extent rather than assumed square.
    coords = np.zeros((rows, cols, 4), dtype=np.float32)
    for r in range(rows):
        y0, y1 = r * cell, min((r + 1) * cell, frame.height)
        for c in range(cols):
            x0, x1 = c * cell, min((c + 1) * cell, frame.width)
            coords[r, c] = (
                (x0 + x1) / 2 / frame.width,
                (y0 + y1) / 2 / frame.height,
                (x1 - x0) / frame.width,
                (y1 - y0) / frame.height,
            )
    return feats, kinds, coords


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
            **(meta or {}),
        },
    )
