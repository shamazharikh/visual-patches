"""Degraded-tier codecs must report what they lack, not emit zeros that look measured."""

import pytest

from vpatch.backends import extract_video
from vpatch.backends.ffmpeg_video import VideoExtractor, capability_for


def test_h264_is_the_full_tier(h264):
    cap = VideoExtractor(h264).capability()
    assert cap.codec == "h264"
    assert cap.motion_vectors and cap.partitions and cap.per_block_qp
    assert not cap.degraded


def test_hevc_exports_nothing(hevc):
    """The only producer of MV side data is ff_print_debug_info2(), called only from
    h264dec.c and mpegvideo_dec.c. No ffmpeg release exports HEVC motion vectors."""
    cap = VideoExtractor(hevc).capability()
    assert cap.codec == "hevc"
    assert not cap.motion_vectors and not cap.partitions and not cap.per_block_qp
    assert cap.degraded


def test_degraded_tier_yields_structure_but_no_fabricated_motion(hevc):
    frames = extract_video(hevc)
    assert len(frames) == 50
    assert "B" in {f.pict_type for f in frames}
    assert all(f.coverage == 0.0 for f in frames)
    assert all(not u.geometry_observed for f in frames for u in f.units)
    assert all(u.mvs == () for f in frames for u in f.units)


@pytest.mark.parametrize("codec,mvs", [("h264", True), ("hevc", False), ("av1", False),
                                       ("vp9", False), ("vp8", False), ("mpeg4", True)])
def test_capability_table(codec, mvs):
    assert capability_for(codec).motion_vectors is mvs


def test_ordering_diverges_and_nothing_is_dropped(h264):
    """Without a decoder drain, 48 of 50 frames come back -- and range(48) is still a
    valid permutation, so an index-only check passes while 4% of the clip is lost."""
    frames = extract_video(h264)
    assert len(frames) == 50
    assert sorted(f.display_index for f in frames) == list(range(50))
    assert sorted(f.decode_index for f in frames) == list(range(50))
    assert [f.decode_index for f in frames] != [f.display_index for f in frames]
    for f in frames:
        if f.pict_type == "I":
            assert f.key_frame or f.coverage == 0.0
