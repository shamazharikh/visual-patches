"""Token budget and pruning, with drop accounting.

M3 measured that `grid` is a feature substitution, not a compression scheme: tokens per
frame are `ceil(W/cell)*ceil(H/cell)` regardless of content. Every token this library
saves is saved here.

**A dropped token is only interpretable if something remains for it to be relative to.**
Dropping every cell whose motion is below a threshold sounds obviously right and is
obviously wrong: on the near-static fixture, 27 of 50 frames contain no moving cell at
all, so the rule deletes more than half the clip and a consumer cannot tell a deleted
frame from a frame that was never there. Absence has to *mean* something.

So pruning here is anchored. Every cell is kept on anchor frames; between anchors only
cells that changed are kept, and the absence of a cell means "unchanged since this
sample's last anchor" -- a statement a consumer can act on. Static regions are
represented by their anchor tokens, which is the answer to "what represents a region once
its zero-motion cells are gone".

Every policy returns a `DropReport` alongside the bundle and stamps it into `meta`. A
policy that cannot say what it dropped is not shippable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from vpatch.types import CodedFrame, PatchBundle


@dataclass(frozen=True)
class DropReport:
    """What a policy removed, in enough detail to audit the result.

    `cells_never_kept` is the one that matters: a cell that appears nowhere in the output
    is a region the consumer has no information about at all, as opposed to one whose
    absence means "unchanged". Anchored policies drive it to zero by construction, and
    the tests assert that.
    """

    policy: str
    kept: int
    dropped: int
    by_rule: dict[str, int] = field(default_factory=dict)
    frames_emptied: int = 0
    cells_never_kept: int = 0
    anchors: list[int] = field(default_factory=list)
    budget_below_anchor_floor: bool = False
    # True when the bundle carried no motion measurement anywhere, so "changed" could
    # never be true and only anchors survived. The kept fraction then looks like a large
    # compression win (0.017 on a 60-frame HEVC clip) while carrying no temporal
    # information at all -- the degraded tier producing an empty result, not a good one.
    no_change_signal: bool = False

    @property
    def total(self) -> int:
        return self.kept + self.dropped

    @property
    def kept_fraction(self) -> float:
        return self.kept / self.total if self.total else 1.0


def keyframe_anchors(frames: list[CodedFrame]) -> list[int]:
    """Anchor on the codec's own keyframes, in display order.

    Preferred over a fixed stride when available: a keyframe is where the encoder decided
    temporal prediction stopped paying, so it is already the frame least well described
    as a delta from its predecessor. Falls back to frame 0 if a stream reports none.
    """
    ordered = sorted(frames, key=lambda f: f.display_index)
    anchors = [t for t, f in enumerate(ordered) if f.key_frame]
    return anchors or [0]


def stride_anchors(n_frames: int, every: int) -> list[int]:
    if every < 1:
        raise ValueError("anchor stride must be >= 1")
    return list(range(0, n_frames, every))


def salience(bundle: PatchBundle, *, qp_weight: float = 0.0) -> np.ndarray:
    """Per-token ranking score for budget enforcement.

    Motion means and motion standard deviations are both in luma pixels, so they add
    without an invented conversion. QP is in a different unit entirely and is weighted 0
    by default -- mixing it in at some plausible-looking scale would be a fabricated
    exchange rate, so a caller who wants it has to name the rate.
    """
    idx = {n: i for i, n in enumerate(bundle.meta["feature_layout"])}
    f = bundle.features

    def mag(a: str, b: str) -> np.ndarray:
        return np.hypot(f[:, idx[a]], f[:, idx[b]])

    score = (mag("l0_dx_mean", "l0_dy_mean") + mag("l1_dx_mean", "l1_dy_mean")
             + mag("l0_dx_std", "l0_dy_std") + mag("l1_dx_std", "l1_dy_std"))
    if qp_weight:
        score = score + qp_weight * f[:, idx["qp_mean"]]
    # A cell built from grid fill was never measured; rank it below anything that was.
    return score * f[:, idx["observed_frac"]]


def _has_motion_signal(bundle: PatchBundle) -> bool:
    """Was motion measured anywhere in this bundle?

    Read from the validity fractions rather than the motion values: a clip that is
    genuinely motionless has l0_frac=1 with zero displacement, while a codec that exports
    no motion at all has l0_frac=0. Those must not be confused -- the first is a
    measurement, the second is its absence.
    """
    idx = {n: i for i, n in enumerate(bundle.meta["feature_layout"])}
    return bool(bundle.features[:, idx["l0_frac"]].any()
                or bundle.features[:, idx["l1_frac"]].any())


def _cell_id(bundle: PatchBundle) -> np.ndarray:
    """Stable spatial identity per token, so 'this cell never appears' is answerable.

    Recovered from each cell's top-left corner, not from its centre. Where `cell` does
    not divide the frame the last row and column are partial, so their centres sit closer
    in than a uniform grid would put them: at cell=28 on a 320-wide frame, scaling the
    centre by the column count maps cells 10 and 11 both to 11 and leaves index 10
    unreachable. That would report nine phantom "never emitted" cells on a bundle where
    anchors had in fact kept every one.
    """
    rows, cols = bundle.meta["grid"]
    width, height = bundle.meta["frame_size"]
    cell = bundle.meta["cell"]
    c = bundle.coords
    x0 = (c[:, 0] - c[:, 2] / 2) * width
    y0 = (c[:, 1] - c[:, 3] / 2) * height
    col = np.clip(np.round(x0 / cell).astype(np.int32), 0, cols - 1)
    row = np.clip(np.round(y0 / cell).astype(np.int32), 0, rows - 1)
    return row * cols + col


def _apply(bundle: PatchBundle, keep: np.ndarray, report: DropReport) -> PatchBundle:
    """Rebuild a bundle from a boolean mask, preserving sample boundaries."""
    bounds = bundle.cu_seqlens
    seq_lens = [int(keep[bounds[i]:bounds[i + 1]].sum()) for i in range(len(bundle.seq_lens))]
    return PatchBundle(
        features=bundle.features[keep],
        coords=bundle.coords[keep],
        times=bundle.times[keep],
        kinds=bundle.kinds[keep],
        seq_lens=seq_lens,
        modality=bundle.modality,
        meta={**bundle.meta, "drop_report": asdict(report)},
    )


def _finish(bundle: PatchBundle, keep: np.ndarray, policy: str,
            by_rule: dict[str, int], anchors: list[int],
            budget_below_anchor_floor: bool = False) -> tuple[PatchBundle, DropReport]:
    cells = _cell_id(bundle)
    n_cells = int(np.prod(bundle.meta["grid"]))
    kept_cells = np.unique(cells[keep])
    kept_times = set(np.unique(bundle.times[keep]).tolist())
    all_times = set(np.unique(bundle.times).tolist())

    report = DropReport(
        policy=policy,
        kept=int(keep.sum()),
        dropped=int((~keep).sum()),
        by_rule=by_rule,
        frames_emptied=len(all_times - kept_times),
        cells_never_kept=n_cells - len(kept_cells),
        anchors=list(anchors),
        budget_below_anchor_floor=budget_below_anchor_floor,
        no_change_signal=not _has_motion_signal(bundle),
    )
    return _apply(bundle, keep, report), report


def keep_all(bundle: PatchBundle) -> tuple[PatchBundle, DropReport]:
    keep = np.ones(len(bundle.features), dtype=bool)
    return _finish(bundle, keep, "keep_all", {}, [])


def anchor_delta(bundle: PatchBundle, *, anchors: list[int],
                 motion_threshold: float = 0.5,
                 qp_delta: float | None = None) -> tuple[PatchBundle, DropReport]:
    """Keep every cell on anchor frames; between them keep only what changed.

    A non-anchor cell survives if its motion exceeds `motion_threshold` luma pixels, if
    its motion is *split* (a non-zero standard deviation means the encoder found more
    than one motion inside the cell, which a mean can hide -- see patchify), or if its QP
    moved by more than `qp_delta` from the preceding anchor.
    """
    idx = {n: i for i, n in enumerate(bundle.meta["feature_layout"])}
    f = bundle.features
    anchor_set = np.isin(bundle.times, np.asarray(anchors, dtype=bundle.times.dtype))

    speed = np.maximum(
        np.hypot(f[:, idx["l0_dx_mean"]], f[:, idx["l0_dy_mean"]]),
        np.hypot(f[:, idx["l1_dx_mean"]], f[:, idx["l1_dy_mean"]]),
    )
    split = np.maximum(
        np.hypot(f[:, idx["l0_dx_std"]], f[:, idx["l0_dy_std"]]),
        np.hypot(f[:, idx["l1_dx_std"]], f[:, idx["l1_dy_std"]]),
    )
    moving = speed >= motion_threshold
    is_split = split >= motion_threshold

    changed = moving | is_split
    qp_changed = np.zeros(len(f), dtype=bool)
    if qp_delta is not None:
        cells = _cell_id(bundle)
        n_cells = int(np.prod(bundle.meta["grid"]))
        # Each token's QP measured against the most recent anchor for the SAME cell.
        ref = np.zeros(n_cells, dtype=np.float32)
        for t in sorted(np.unique(bundle.times).tolist()):
            sel = bundle.times == t
            if t in anchors:
                ref[cells[sel]] = f[sel, idx["qp_mean"]]
            else:
                qp_changed[sel] = np.abs(f[sel, idx["qp_mean"]] - ref[cells[sel]]) > qp_delta
        changed = changed | qp_changed

    keep = anchor_set | changed
    dropped = ~keep
    by_rule = {
        "unchanged_since_anchor": int(dropped.sum()),
        "kept_anchor": int(anchor_set.sum()),
        "kept_moving": int((moving & ~anchor_set).sum()),
        "kept_split_motion": int((is_split & ~moving & ~anchor_set).sum()),
    }
    if qp_delta is not None:
        # Against `moving`/`is_split`, not against `changed` -- `changed` already
        # absorbed qp_changed, so testing it there would report 0 forever.
        by_rule["kept_qp_shift"] = int(
            (qp_changed & ~anchor_set & ~moving & ~is_split).sum()
        )
    return _finish(bundle, keep, "anchor_delta", by_rule, anchors)


def budget(bundle: PatchBundle, max_tokens: int, *, anchors: list[int] | None = None,
           qp_weight: float = 0.0) -> tuple[PatchBundle, DropReport]:
    """Hard token cap. Never returns more than `max_tokens`.

    Anchor tokens outrank every non-anchor token, so a budget bites into deltas before it
    bites into the frames that make deltas interpretable. If the budget is smaller than
    the anchor floor itself, anchors are dropped by score -- and `budget_below_anchor_floor`
    is set, because at that point the "absence means unchanged" contract no longer holds
    and the caller has to know.
    """
    if max_tokens < 0:
        raise ValueError("max_tokens must be >= 0")
    n = len(bundle.features)
    anchors = anchors or []
    score = salience(bundle, qp_weight=qp_weight)
    is_anchor = np.isin(bundle.times, np.asarray(anchors, dtype=bundle.times.dtype))

    if n <= max_tokens:
        return _finish(bundle, np.ones(n, dtype=bool), "budget",
                       {"within_budget": n}, anchors)

    below_floor = bool(is_anchor.sum() > max_tokens)
    # Rank anchors above everything else by lifting them past the score range, rather
    # than sorting twice: ties inside each tier still resolve by salience.
    rank = score + is_anchor * (float(score.max()) + 1.0)
    # Stable order so the cap is a pure function of the bundle, not of sort internals.
    order = np.lexsort((np.arange(n), -rank))
    keep = np.zeros(n, dtype=bool)
    keep[order[:max_tokens]] = True

    dropped = ~keep
    by_rule = {
        "over_budget": int(dropped.sum()),
        "dropped_anchor": int((dropped & is_anchor).sum()),
        "dropped_delta": int((dropped & ~is_anchor).sum()),
    }
    return _finish(bundle, keep, "budget", by_rule, anchors, below_floor)


def prune(bundle: PatchBundle, *, anchors: list[int], max_tokens: int | None = None,
          motion_threshold: float = 0.5, qp_delta: float | None = None,
          qp_weight: float = 0.0) -> tuple[PatchBundle, DropReport]:
    """anchor_delta, then a hard cap. The order matters.

    Delta pruning first, budget second: the budget is a last-resort truncation that drops
    tokens purely by score, with no statement about what the absence means, so it should
    only ever see what delta pruning could not already remove for a stated reason.
    """
    out, rep = anchor_delta(bundle, anchors=anchors, motion_threshold=motion_threshold,
                            qp_delta=qp_delta)
    if max_tokens is None or len(out.features) <= max_tokens:
        return out, rep
    out2, rep2 = budget(out, max_tokens, anchors=anchors, qp_weight=qp_weight)
    merged = DropReport(
        policy="prune",
        kept=rep2.kept,
        dropped=rep.dropped + rep2.dropped,
        by_rule={**rep.by_rule, **rep2.by_rule},
        frames_emptied=rep2.frames_emptied,
        cells_never_kept=rep2.cells_never_kept,
        anchors=anchors,
        budget_below_anchor_floor=rep2.budget_below_anchor_floor,
        no_change_signal=rep.no_change_signal,
    )
    out2.meta = {**out2.meta, "drop_report": asdict(merged)}
    return out2, merged
