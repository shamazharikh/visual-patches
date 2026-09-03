"""The degraded tier, the typed rejections, and the resource caps.

The theme: an input this library cannot describe honestly must produce a named exception
or an explicit "not available", never a plausible-looking number.
"""

import itertools

import numpy as np
import pytest

from vpatch.backends.ffmpeg_video import (
    VideoExtractor,
    capability_for,
    extract_video,
)
from vpatch.patchify import KIND_INTRA, patchify_grid
from vpatch.sampling import anchor_delta, keyframe_anchors
from vpatch.types import (
    ResolutionChanged,
    ResourceLimitExceeded,
    UnsupportedCodecFeature,
)

DEGRADED = ["hevc", "av1", "vp8"]


@pytest.mark.parametrize("codec", DEGRADED)
def test_degraded_codecs_declare_what_they_lack(codec):
    cap = capability_for(codec)
    assert not cap.motion_vectors
    assert not cap.partitions
    assert not cap.per_block_qp
    assert cap.degraded


@pytest.mark.parametrize("fixture", DEGRADED)
def test_degraded_codecs_yield_structure_but_never_fabricated_motion(request, fixture):
    """Frame types and ordering still work; motion does not appear from nowhere."""
    frames = extract_video(request.getfixturevalue(fixture), pixels=False)
    assert frames
    assert all(f.coverage == 0.0 for f in frames)
    assert not any(u.mvs for f in frames for u in f.units)
    assert all(not u.geometry_observed for f in frames for u in f.units)
    assert {f.pict_type for f in frames} <= {"I", "P", "B", "S", "SI", "SP", "BI", "NONE"}

    bundle = patchify_grid(frames)
    assert set(np.unique(bundle.kinds)) == {KIND_INTRA}
    # Every cell is grid fill, so the whole bundle is flagged as unobserved.
    idx = bundle.meta["feature_layout"].index("observed_frac")
    assert not bundle.features[:, idx].any()

    _, report = anchor_delta(bundle, anchors=keyframe_anchors(frames))
    assert report.no_change_signal


def test_interlaced_input_is_rejected_rather_than_halved(interlaced):
    """The guard is necessary, not theoretical: this clip DOES export motion vectors.

    ffmpeg's field-to-frame correction (`my *= 2`) exists only in the IS_16X8/IS_8X16
    branches, so 16x16 and 8x8 macroblocks would silently carry half-magnitude vertical
    motion -- a wrong number, not a missing one.
    """
    with pytest.raises(UnsupportedCodecFeature, match="interlaced"):
        extract_video(interlaced, pixels=False)


def test_the_interlaced_fixture_would_otherwise_produce_motion(interlaced):
    """Pins the premise of the test above, so the guard cannot become a no-op silently."""
    import av
    from av.codec.context import Flags2

    with av.open(interlaced) as container:
        stream = container.streams.video[0]
        stream.codec_context.flags2 |= Flags2.export_mvs
        stream.codec_context.options = {"export_side_data": "mvs+venc_params"}
        frames = list(itertools.islice(container.decode(stream), 6))
    assert all(f.interlaced_frame for f in frames)
    assert sum(f.side_data.get("MOTION_VECTORS") is not None for f in frames) > 0


def test_mid_stream_resolution_change_is_rejected(reschange):
    """One normalized coordinate space cannot describe two coded sizes."""
    with pytest.raises(ResolutionChanged):
        extract_video(reschange, pixels=False)


def test_ten_bit_input_extracts_and_reports_its_depth(h264_10bit):
    """QP range shifts by 6*(bit_depth-8), so depth has to travel with the values."""
    frames = extract_video(h264_10bit, pixels=False)
    assert all(f.bit_depth == 10 for f in frames)
    assert any(f.coverage > 0 for f in frames)
    qps = [u.qp for f in frames for u in f.units if u.qp is not None]
    assert qps
    # A 10-bit stream can legitimately use QP below the 8-bit floor of 0.
    assert min(qps) >= -12


def test_max_pixels_is_a_resource_limit_not_a_codec_complaint(h264):
    with pytest.raises(ResourceLimitExceeded, match="max_pixels"):
        VideoExtractor(h264, pixels=False, max_pixels=1000).extract()


def test_max_decode_seconds_bounds_the_whole_read(h264):
    with pytest.raises(ResourceLimitExceeded, match="max_decode_seconds"):
        VideoExtractor(h264, pixels=False, max_decode_seconds=0.0).extract()


def test_a_generous_deadline_does_not_fire(h264):
    frames = VideoExtractor(h264, pixels=False, max_decode_seconds=600).extract()
    assert len(frames) == 50


def test_max_frames_truncates_without_raising(h264):
    """A caller-set frame budget is a normal request, not an error condition."""
    frames = VideoExtractor(h264, pixels=False, max_frames=7).extract()
    assert len(frames) == 7


def test_probe_separates_declared_capability_from_observed(vp9_noaq, vp9):
    """VP9 declares partitions; only a segmentation-enabled bitstream delivers them."""
    assert VideoExtractor(vp9_noaq, pixels=False).capability().partitions
    assert VideoExtractor(vp9_noaq, pixels=False).capability(probe_frames=8).observed is False
    assert VideoExtractor(vp9, pixels=False).capability(probe_frames=8).observed is True


@pytest.mark.parametrize("fixture", DEGRADED)
def test_probe_reports_degraded_codecs_as_unobserved(request, fixture):
    ex = VideoExtractor(request.getfixturevalue(fixture), pixels=False)
    assert ex.capability(probe_frames=8).observed is False


def test_capability_is_keyed_on_the_codec_not_the_decoder(av1, h264):
    """`codec_context.name` is the decoder: AV1 reads back as "libdav1d".

    A build linked against libopenh264 would report "libopenh264" for ordinary H.264 and
    silently demote a fully capable file to the degraded tier.
    """
    import av

    from vpatch.backends.ffmpeg_video import canonical_codec

    with av.open(av1) as container:
        ctx = container.streams.video[0].codec_context
        assert ctx.name != "av1", "fixture no longer exercises the decoder/codec split"
        assert canonical_codec(ctx) == "av1"
    with av.open(h264) as container:
        assert canonical_codec(container.streams.video[0].codec_context) == "h264"
