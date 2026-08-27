"""Gates on the grid aggregation rule and the token-count claim it exists to test."""

import math

import numpy as np
import pytest

from vpatch.backends.ffmpeg_video import extract_video
from vpatch.partition import grid_shape, overlap_pixels
from vpatch.patchify import (
    D,
    FEATURE_LAYOUT,
    KIND_INTER,
    KIND_INTRA,
    KIND_UNKNOWN,
    frame_features,
    patchify_grid,
)
from vpatch.types import CodedFrame, CodedUnit, MotionVector, UnitKind

IDX = {n: i for i, n in enumerate(FEATURE_LAYOUT)}


def _frame(units, w=16, h=16, **kw):
    return CodedFrame(
        decode_index=0, display_index=0, pict_type="P", key_frame=False,
        width=w, height=h, bit_depth=8, pix_fmt="yuv420p", units=units, **kw
    )


def _unit(x, y, w, h, mvs=(), kind=UnitKind.INTER, observed=True, qp=None):
    return CodedUnit(x=x, y=y, w=w, h=h, kind=kind, geometry_observed=observed,
                     mvs=tuple(MotionVector(dx, dy, li) for dx, dy, li in mvs), qp=qp)


def test_mean_alone_would_call_a_split_macroblock_static():
    """The reason every motion channel carries a std.

    A macroblock split into four 8x8 partitions moving (+4,0), (-4,0), (+4,0), (-4,0)
    has mean motion exactly zero -- indistinguishable, under a plain mean, from a cell
    the codec found completely static. The std is what separates them.
    """
    split = _frame([
        _unit(0, 0, 8, 8, [(+4.0, 0.0, 0)]), _unit(8, 0, 8, 8, [(-4.0, 0.0, 0)]),
        _unit(0, 8, 8, 8, [(+4.0, 0.0, 0)]), _unit(8, 8, 8, 8, [(-4.0, 0.0, 0)]),
    ])
    static = _frame([_unit(0, 0, 16, 16, [(0.0, 0.0, 0)])])

    f_split = frame_features(split)[0][0, 0]
    f_static = frame_features(static)[0][0, 0]

    assert f_split[IDX["l0_dx_mean"]] == pytest.approx(0.0, abs=1e-5)
    assert f_static[IDX["l0_dx_mean"]] == pytest.approx(0.0, abs=1e-5)
    # The means agree; only the second moment tells them apart.
    assert f_split[IDX["l0_dx_std"]] == pytest.approx(4.0, abs=1e-4)
    assert f_static[IDX["l0_dx_std"]] == pytest.approx(0.0, abs=1e-6)


def test_aggregation_is_weighted_by_overlap_area():
    """A block covering a quarter of a cell contributes a quarter of the weight."""
    frame = _frame([
        _unit(0, 0, 8, 8, [(8.0, 0.0, 0)]),      # 64 px of the 256 px cell
        _unit(8, 0, 8, 8, [(0.0, 0.0, 0)]),
        _unit(0, 8, 16, 8, [(0.0, 0.0, 0)]),     # 128 px
    ])
    f = frame_features(frame)[0][0, 0]
    assert f[IDX["l0_dx_mean"]] == pytest.approx(8.0 * 64 / 256, abs=1e-4)


def test_a_superblock_spans_every_cell_it_covers():
    """One 64x64 VP9 block must reach all sixteen 16x16 cells, not just its origin."""
    frame = _frame([_unit(0, 0, 64, 64, [(2.0, 0.0, 0)])], w=64, h=64)
    f, _, _ = frame_features(frame)
    assert f.shape[:2] == (4, 4)
    assert np.allclose(f[..., IDX["l0_dx_mean"]], 2.0)
    assert np.allclose(f[..., IDX["log2_area_mean"]], math.log2(64 * 64))


def test_measured_zero_motion_is_distinguishable_from_no_measurement():
    """`l0_frac` is the channel that keeps a zero from being a hole."""
    measured = frame_features(_frame([_unit(0, 0, 16, 16, [(0.0, 0.0, 0)])]))[0][0, 0]
    absent = frame_features(_frame([
        _unit(0, 0, 16, 16, kind=UnitKind.INTRA, observed=False)
    ]))[0][0, 0]

    assert measured[IDX["l0_dx_mean"]] == absent[IDX["l0_dx_mean"]] == 0.0
    assert measured[IDX["l0_frac"]] == pytest.approx(1.0)
    assert absent[IDX["l0_frac"]] == 0.0
    assert measured[IDX["observed_frac"]] == pytest.approx(1.0)
    assert absent[IDX["observed_frac"]] == 0.0


def test_kinds_do_not_guess_on_a_codec_with_no_intra_inter_signal(vp9):
    """VP9 exports geometry with no prediction-mode signal; cells stay UNKNOWN."""
    bundle = patchify_grid(extract_video(vp9, pixels=False))
    assert set(np.unique(bundle.kinds)) == {KIND_UNKNOWN}
    motion_channels = [IDX[n] for n in FEATURE_LAYOUT
                       if n.startswith(("l0_", "l1_", "bipred", "inter_"))]
    # Not "approximately zero": no motion was exported, so none may appear.
    assert not bundle.features[:, motion_channels].any()


def test_h264_cells_are_labelled_and_carry_motion(h264):
    bundle = patchify_grid(extract_video(h264, pixels=False))
    assert set(np.unique(bundle.kinds)) <= {KIND_INTRA, KIND_INTER}
    assert KIND_INTER in np.unique(bundle.kinds)
    assert np.abs(bundle.features[:, IDX["l0_dx_mean"]]).max() > 0


def test_coords_are_normalised_to_the_visible_frame(odd):
    """250x170 is not a multiple of 16: the last cell column is 10px, not 16."""
    frames = extract_video(odd, pixels=False)
    bundle = patchify_grid(frames)
    c = bundle.coords
    assert c.min() >= 0.0 and c.max() <= 1.0
    # Centre +/- half-extent must stay inside the frame -- proof nothing normalised
    # against padded coded space, where the bottom row overhangs.
    assert (c[:, 0] + c[:, 2] / 2 <= 1.0 + 1e-6).all()
    assert (c[:, 1] + c[:, 3] / 2 <= 1.0 + 1e-6).all()
    widths = np.unique(np.round(c[:, 2] * 250).astype(int))
    assert set(widths) == {16, 10}  # 250 = 15*16 + 10


def test_token_count_is_exactly_cells_times_frames(h264):
    frames = extract_video(h264, pixels=False)
    rows, cols = grid_shape(frames[0].width, frames[0].height)
    bundle = patchify_grid(frames)
    assert len(bundle.features) == rows * cols * len(frames)
    assert bundle.features.shape[1] == D
    assert bundle.seq_lens == [len(bundle.features)]
    assert bundle.cu_seqlens.tolist() == [0, len(bundle.features)]


def test_times_are_display_order_not_decode_order(h264):
    """B-frames make these differ; a bundle ordered by decode index would be scrambled."""
    frames = extract_video(h264, pixels=False)
    assert [f.decode_index for f in frames] != [f.display_index for f in frames]
    bundle = patchify_grid(frames)
    assert bundle.times.tolist() == sorted(bundle.times.tolist())
    assert bundle.times.max() == len(frames) - 1


@pytest.mark.parametrize("cell", [16, 28, 32])
def test_grid_token_count_is_structural_not_content_dependent(h264, pan, cell):
    """The gate result, as an assertion.

    Tokens per frame are ceil(W/cell)*ceil(H/cell) -- a function of resolution and cell
    size alone. Two clips of identical geometry and very different motion produce
    identical counts, so `grid` cannot be a token-reduction scheme. Any reduction has to
    come from an explicit pruning policy (M4), and this test is what stops a future
    change from quietly reintroducing the "tokens follow bitrate" claim.
    """
    a = patchify_grid(extract_video(h264, pixels=False), cell=cell)
    b = patchify_grid(extract_video(pan, pixels=False), cell=cell)
    assert a.meta["grid"] == b.meta["grid"]
    per_frame_a = len(a.features) / a.meta["n_frames"]
    per_frame_b = len(b.features) / b.meta["n_frames"]
    assert per_frame_a == per_frame_b
    rows, cols = grid_shape(320, 240, cell)
    assert per_frame_a == rows * cols


@pytest.mark.parametrize("path_fixture", ["h264", "vp9"])
def test_patchify_is_invariant_to_thread_count(request, path_fixture):
    """Extraction purity has to survive aggregation, which is where a stray order shows."""
    path = request.getfixturevalue(path_fixture)
    one = patchify_grid(extract_video(path, pixels=False, thread_count=1))
    four = patchify_grid(extract_video(path, pixels=False, thread_count=4))
    assert one.features.tobytes() == four.features.tobytes()
    assert one.kinds.tobytes() == four.kinds.tobytes()
    assert one.coords.tobytes() == four.coords.tobytes()


def test_vp9_tiling_has_no_overlapping_pixels(vp9):
    """The independent oracle for `coverage`: an exact tiling double-claims nothing."""
    for frame in extract_video(vp9, pixels=False, fill_holes=False)[:5]:
        assert overlap_pixels(frame.units, frame.width, frame.height) == 0
        assert frame.coverage == pytest.approx(1.0)
