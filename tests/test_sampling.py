"""Gates on pruning: what it removes, what it promises about what it kept."""

import numpy as np
import pytest

from vpatch.backends.ffmpeg_video import extract_video
from vpatch.patchify import FEATURE_LAYOUT, patchify_grid
from vpatch.sampling import (
    _cell_id,
    anchor_delta,
    budget,
    keep_all,
    keyframe_anchors,
    prune,
    salience,
    stride_anchors,
)

IDX = {n: i for i, n in enumerate(FEATURE_LAYOUT)}
FIXTURES = ["h264_b3.mp4", "static_box.mp4", "pan.mp4", "odd_250x170.mp4"]


def _bundle(request, name, cell=16):
    frames = extract_video(f"tests/assets/{name}", pixels=False)
    return frames, patchify_grid(frames, cell=cell)


def test_naive_motion_pruning_deletes_a_near_static_clip(static_box):
    """Why anchors exist, stated as the counterexample they answer.

    Dropping every cell below a motion threshold is the obvious policy. On a scene that
    is static except for one small moving object it removes 27 of 50 frames outright and
    leaves most of the grid with no token anywhere in the clip -- so a consumer cannot
    distinguish "this region did not change" from "this region was never described".
    """
    frames = extract_video(static_box, pixels=False)
    bundle = patchify_grid(frames)
    f = bundle.features
    speed = np.hypot(f[:, IDX["l0_dx_mean"]], f[:, IDX["l0_dy_mean"]])
    naive = speed >= 0.5

    emptied = sum(1 for t in np.unique(bundle.times) if not naive[bundle.times == t].any())
    n_cells = int(np.prod(bundle.meta["grid"]))
    lost = n_cells - len(np.unique(_cell_id(bundle)[naive]))
    assert emptied > len(frames) // 2, "fixture no longer exercises the failure"
    assert lost > 0

    # The anchored policy on the same clip: prunes hard, loses no region.
    _, report = anchor_delta(bundle, anchors=keyframe_anchors(frames))
    assert report.kept_fraction < 0.25
    assert report.cells_never_kept == 0


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("cell", [16, 28, 32])
def test_anchors_guarantee_every_cell_is_emitted(request, name, cell):
    """The contract that makes an absent token mean 'unchanged' rather than 'unknown'."""
    frames, bundle = _bundle(request, name, cell)
    _, report = anchor_delta(bundle, anchors=keyframe_anchors(frames))
    assert report.cells_never_kept == 0


def test_cell_id_survives_a_partial_last_column(odd):
    """Regression: recovering a cell index by scaling its CENTRE collides.

    Where the cell size does not divide the frame, the final row and column are narrower,
    so their centres sit closer in than a uniform grid puts them -- at cell=28 on a
    320-wide frame, cells 10 and 11 both map to 11 and index 10 becomes unreachable. The
    accounting then reports phantom lost cells on a bundle that lost nothing.
    """
    frames = extract_video(odd, pixels=False)
    for cell in (16, 28, 32):
        bundle = patchify_grid(frames, cell=cell)
        ids = _cell_id(bundle)
        rows, cols = bundle.meta["grid"]
        # Every grid position must be reachable, exactly once per frame.
        assert set(np.unique(ids).tolist()) == set(range(rows * cols))
        first = ids[bundle.times == 0]
        assert len(first) == len(set(first.tolist())) == rows * cols


def test_budget_is_a_hard_cap_at_1080p(hd):
    frames = extract_video(hd, pixels=False)
    bundle = patchify_grid(frames)
    assert len(bundle.features) > 10_000
    for cap in (0, 1, 997, 10_000):
        out, report = budget(bundle, cap, anchors=keyframe_anchors(frames))
        assert len(out.features) == report.kept <= cap
        assert sum(out.seq_lens) == len(out.features)


def test_budget_spends_on_anchors_first(h264):
    """A budget must bite into deltas before the frames that make deltas readable."""
    frames = extract_video(h264, pixels=False)
    bundle = patchify_grid(frames)
    anchors = keyframe_anchors(frames)
    n_anchor_tokens = int(np.isin(bundle.times, anchors).sum())

    out, report = budget(bundle, n_anchor_tokens, anchors=anchors)
    assert report.dropped > 0
    assert report.by_rule["dropped_anchor"] == 0
    assert set(np.unique(out.times).tolist()) == set(anchors)
    assert not report.budget_below_anchor_floor


def test_a_budget_below_the_anchor_floor_says_so(h264):
    """Past this point 'absence means unchanged' is false, and silence would be a lie."""
    frames = extract_video(h264, pixels=False)
    bundle = patchify_grid(frames)
    anchors = keyframe_anchors(frames)
    n_anchor_tokens = int(np.isin(bundle.times, anchors).sum())

    _, report = budget(bundle, n_anchor_tokens // 2, anchors=anchors)
    assert report.budget_below_anchor_floor
    assert report.by_rule["dropped_anchor"] > 0


def test_budget_is_a_pure_function_of_the_bundle(h264):
    frames = extract_video(h264, pixels=False)
    bundle = patchify_grid(frames)
    a, _ = budget(bundle, 3000, anchors=keyframe_anchors(frames))
    b, _ = budget(bundle, 3000, anchors=keyframe_anchors(frames))
    assert a.features.tobytes() == b.features.tobytes()
    assert a.coords.tobytes() == b.coords.tobytes()


def test_grid_fill_never_outranks_a_real_measurement(h264):
    """Salience is gated on observed_frac: inferred geometry is not evidence."""
    frames = extract_video(h264, pixels=False)
    bundle = patchify_grid(frames)
    score = salience(bundle)
    unobserved = bundle.features[:, IDX["observed_frac"]] == 0
    assert unobserved.any()
    assert not score[unobserved].any()


def test_no_motion_export_means_no_fabricated_pruning(vp9):
    """VP9 exports geometry but no motion; nothing may 'change' by a motion rule."""
    frames = extract_video(vp9, pixels=False)
    bundle = patchify_grid(frames)
    anchors = keyframe_anchors(frames)
    _, report = anchor_delta(bundle, anchors=anchors)
    assert report.by_rule["kept_moving"] == 0
    assert report.by_rule["kept_split_motion"] == 0
    assert report.kept == report.by_rule["kept_anchor"]


@pytest.mark.parametrize("name", FIXTURES)
def test_drop_accounting_balances(request, name):
    frames, bundle = _bundle(request, name)
    out, report = anchor_delta(bundle, anchors=keyframe_anchors(frames))
    assert report.kept + report.dropped == len(bundle.features)
    assert report.kept == len(out.features) == sum(out.seq_lens)
    assert out.meta["drop_report"]["policy"] == "anchor_delta"
    assert out.meta["drop_report"]["kept"] == report.kept


def test_qp_delta_is_off_by_default_because_rate_control_dominates(static_box):
    """Documents the measurement behind the default.

    QP is re-chosen by rate control every frame, so on a near-static clip a QP-change
    rule keeps almost everything: it is measuring the encoder's bit allocation, not the
    scene. Motion alone prunes the same clip to ~12%.
    """
    frames = extract_video(static_box, pixels=False)
    bundle = patchify_grid(frames)
    anchors = keyframe_anchors(frames)
    _, off = anchor_delta(bundle, anchors=anchors)
    _, on = anchor_delta(bundle, anchors=anchors, qp_delta=2.0)
    assert off.kept_fraction < 0.25
    assert on.kept_fraction > 0.9


def test_prune_composes_delta_then_cap(h264):
    frames = extract_video(h264, pixels=False)
    bundle = patchify_grid(frames)
    anchors = keyframe_anchors(frames)
    out, report = prune(bundle, anchors=anchors, max_tokens=2000)
    assert len(out.features) == report.kept <= 2000
    assert report.policy == "prune"
    # Both stages' reasons survive into one report.
    assert "unchanged_since_anchor" in report.by_rule
    assert "over_budget" in report.by_rule
    assert report.kept + report.dropped == len(bundle.features)
    assert out.meta["drop_report"]["kept"] == report.kept


def test_prune_skips_the_cap_when_delta_pruning_already_fits(static_box):
    frames = extract_video(static_box, pixels=False)
    bundle = patchify_grid(frames)
    _, report = prune(bundle, anchors=keyframe_anchors(frames), max_tokens=1_000_000)
    assert report.policy == "anchor_delta"
    assert "over_budget" not in report.by_rule


def test_keep_all_is_the_identity(h264):
    frames = extract_video(h264, pixels=False)
    bundle = patchify_grid(frames)
    out, report = keep_all(bundle)
    assert report.dropped == 0
    assert out.features.tobytes() == bundle.features.tobytes()
    assert out.seq_lens == bundle.seq_lens


def test_stride_anchors_cover_the_clip(h264):
    frames = extract_video(h264, pixels=False)
    bundle = patchify_grid(frames)
    _, report = anchor_delta(bundle, anchors=stride_anchors(len(frames), 10))
    assert report.cells_never_kept == 0
    with pytest.raises(ValueError):
        stride_anchors(10, 0)
