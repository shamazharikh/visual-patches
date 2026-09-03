"""VP9's superblock quadtree -- the non-uniform partition tree, via the public av API.

H.264's MV records give two depths, 73-84% of units at one size, zero coverage on I-frames,
and 1.70x area overshoot from bi-prediction. VP9's enc-params blocks give four depths and an
exact tiling. This suite pins the properties that make it worth having.
"""

from collections import Counter

from vpatch.backends import extract_video
from vpatch.backends.ffmpeg_video import VideoExtractor


def test_vp9_declares_partitions_without_motion(vp9):
    """The three capability axes are independent: VP9 has geometry but no motion."""
    cap = VideoExtractor(vp9).capability()
    assert cap.partitions and cap.per_block_qp
    assert not cap.motion_vectors
    assert not cap.degraded


def test_partition_tree_has_four_depths(vp9):
    units = [u for f in extract_video(vp9) for u in f.observed_units]
    shapes = Counter((u.w, u.h) for u in units)
    assert set(shapes) == {(8, 8), (16, 16), (32, 32), (64, 64)}
    assert len(units) > 10_000
    # Genuinely non-uniform: no single size dominates the way 16x16 does in H.264.
    assert max(shapes.values()) / len(units) < 0.75


def test_tiling_is_exact_and_non_overlapping(vp9):
    """Every pixel covered exactly once, on every frame -- including I-frames."""
    import numpy as np

    for f in extract_video(vp9):
        mask = np.zeros((f.height, f.width), np.int32)
        for u in f.observed_units:
            mask[u.y:u.y + u.h, u.x:u.x + u.w] += 1
        assert mask.min() == 1, "gap in the partition tree"
        assert mask.max() == 1, "overlapping blocks in the partition tree"
        assert f.coverage == 1.0


def test_no_fabricated_motion(vp9):
    assert all(u.mvs == () for f in extract_video(vp9) for u in f.units)


def test_kind_is_none_not_guessed(vp9):
    """AVVideoBlockParams carries no intra/inter signal, so kind must not claim one."""
    assert all(u.kind is None for f in extract_video(vp9) for u in f.observed_units)


def test_per_block_qp_varies(vp9):
    qps = {u.qp for f in extract_video(vp9) for u in f.observed_units}
    assert len(qps) > 10, "per-block delta_qp is not being applied"


def test_declared_capability_is_not_observed_capability(vp9, vp9_noaq):
    """Block export is gated on segmentation. A codec-name table cannot see that, so the
    probe trial-decodes: same codec, same declared capability, opposite reality."""
    assert VideoExtractor(vp9).capability().partitions
    assert VideoExtractor(vp9_noaq).capability().partitions

    assert VideoExtractor(vp9, pixels=False).capability(probe_frames=5).observed is True
    assert VideoExtractor(vp9_noaq, pixels=False).capability(probe_frames=5).observed is False


def test_ungated_vp9_yields_no_fabricated_geometry(vp9_noaq):
    frames = extract_video(vp9_noaq)
    assert all(f.coverage == 0.0 for f in frames)
    assert all(not u.geometry_observed for f in frames for u in f.units)
