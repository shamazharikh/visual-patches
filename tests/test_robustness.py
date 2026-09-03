"""Gates on the conditions real footage produces that synthetic fixtures do not.

Every case here was found by running the library over a corpus of surveillance clips,
not by reading the spec.
"""

import numpy as np
import pytest

from vpatch.backends.ffmpeg_video import (
    VideoExtractor,
    extract_video,
    open_container,
    rank_packets,
)
from vpatch.patchify import patchify_grid
from vpatch.sampling import anchor_delta, keyframe_anchors
from vpatch.types import CorruptBitstream


def test_duplicate_pts_does_not_undercount_packets():
    """PTS is not unique in real streams, and a pts-keyed dict silently loses the copies.

    A surveillance clip in the wild opens with two packets both stamped pts=77. Deriving
    the packet count from the size of a pts->rank dict undercounted by exactly one and
    turned a healthy 84-frame clip into a spurious "decoder was not drained" failure.
    """
    packets = [(77, 77), (77, 77), (243, 243), (410, 410)]
    ranks = rank_packets(packets)
    assert ranks[77] == [0, 1], "both copies must keep a distinct decode rank"
    assert sum(len(v) for v in ranks.values()) == len(packets)


def test_decode_ranks_follow_dts_not_pts():
    """Reordered stream: display order is by PTS, decode order by DTS."""
    packets = [(0, -1024), (2048, -512), (1024, 0), (512, 512)]
    ranks = rank_packets(packets)
    assert [ranks[p][0] for p, _ in packets] == [0, 1, 2, 3]
    # And the display order is genuinely different from the decode order.
    assert sorted(packets, key=lambda pd: pd[0]) != packets


def test_packets_without_timestamps_are_not_given_an_index():
    ranks = rank_packets([(None, 0), (10, 1), (None, 2)])
    assert set(ranks) == {10}


def test_fixtures_with_unique_timestamps_are_not_flagged_ambiguous(h264, pan):
    for path in (h264, pan):
        frames = extract_video(path, pixels=False)
        assert not any(f.order_ambiguous for f in frames)
        assert sorted(f.decode_index for f in frames) == list(range(len(frames)))


def test_a_truncated_container_raises_a_typed_error(h264, tmp_path):
    """An MP4's moov atom is at the end, so a truncated file fails at OPEN, not decode.

    Letting PyAV's InvalidDataError escape would make callers catch a dependency's
    exception type for a condition this library already names.
    """
    data = (open(h264, "rb").read())
    for fraction in (0.6, 0.85):
        cut = tmp_path / f"cut_{fraction}.mp4"
        cut.write_bytes(data[: int(len(data) * fraction)])
        with pytest.raises(CorruptBitstream):
            extract_video(str(cut), pixels=False)
        with pytest.raises(CorruptBitstream):
            open_container(str(cut))


def test_a_non_video_file_raises_a_typed_error(tmp_path):
    junk = tmp_path / "notavideo.mp4"
    junk.write_bytes(b"this is not a bitstream" * 100)
    with pytest.raises(CorruptBitstream):
        extract_video(str(junk), pixels=False)


def test_a_degraded_codec_reports_an_empty_result_not_a_compression_win(hevc):
    """HEVC exports no motion, so nothing can ever count as changed.

    Only anchors survive, which looks like a large kept-fraction win while carrying no
    temporal information at all. The report has to say which one it is.
    """
    frames = extract_video(hevc, pixels=False)
    bundle = patchify_grid(frames)
    _, report = anchor_delta(bundle, anchors=keyframe_anchors(frames))
    assert report.no_change_signal
    assert report.kept_fraction < 0.5  # would read as a win without the flag
    assert report.kept == report.by_rule["kept_anchor"]


def test_a_motionless_clip_is_not_confused_with_a_codec_that_exports_no_motion(static_box):
    """l0_frac separates 'measured, and it was zero' from 'never measured'."""
    frames = extract_video(static_box, pixels=False)
    bundle = patchify_grid(frames)
    _, report = anchor_delta(bundle, anchors=keyframe_anchors(frames))
    assert not report.no_change_signal
    assert report.by_rule["kept_moving"] > 0


def _reference_features(frame, cell):
    """Deliberately naive per-unit, per-cell aggregation to check the vectorised path."""
    from vpatch.partition import grid_shape
    from vpatch.patchify import FEATURE_LAYOUT, D

    idx = {n: i for i, n in enumerate(FEATURE_LAYOUT)}
    rows, cols = grid_shape(frame.width, frame.height, cell)
    w_acc = np.zeros((rows, cols))
    s_acc = np.zeros((rows, cols))
    ss_acc = np.zeros((rows, cols))
    area = np.zeros((rows, cols))
    for u in frame.units:
        for r in range(rows):
            for c in range(cols):
                oy = min(u.y + u.h, min((r + 1) * cell, frame.height)) - max(u.y, r * cell)
                ox = min(u.x + u.w, min((c + 1) * cell, frame.width)) - max(u.x, c * cell)
                if ox <= 0 or oy <= 0:
                    continue
                a = ox * oy
                area[r, c] += a
                for m in u.mvs:
                    if m.list_idx == 0:
                        w_acc[r, c] += a
                        s_acc[r, c] += a * m.dx
                        ss_acc[r, c] += a * m.dx * m.dx
    wsafe = np.maximum(w_acc, 1e-12)
    mean = np.where(w_acc == 0, 0.0, s_acc / wsafe)
    std = np.where(w_acc == 0, 0.0,
                   np.sqrt(np.maximum(ss_acc / wsafe - mean * mean, 0.0)))
    return mean, std, idx["l0_dx_mean"], idx["l0_dx_std"], D


@pytest.mark.parametrize("cell", [16, 28])
def test_vectorised_aggregation_matches_a_naive_reference(h264, cell):
    """The fast path is an optimisation, so it has to agree with the obvious one."""
    from vpatch.patchify import frame_features

    frames = extract_video(h264, pixels=False)
    for frame in frames[:4]:
        mean, std, i_mean, i_std, _ = _reference_features(frame, cell)
        feats, _, _ = frame_features(frame, cell)
        assert np.allclose(feats[..., i_mean], mean, atol=1e-4)
        assert np.allclose(feats[..., i_std], std, atol=1e-4)


def test_extraction_stays_within_the_hard_unit_bound(hd):
    """max_units_per_frame is exact, not an estimate: 8x8 is the finest ffmpeg emits."""
    from vpatch.backends.ffmpeg_video import max_units_per_frame

    frames = VideoExtractor(hd, pixels=False, fill_holes=False, max_frames=5).extract()
    bound = max_units_per_frame(frames[0].width, frames[0].height)
    for f in frames:
        assert len(f.units) <= bound


def test_exif_does_not_make_a_photograph_unreadable(exif_jpeg):
    """An ordinary camera JPEG must not fail the way a corrupt one does.

    PyAV maps every side-data entry through an IntEnum while building the container, so
    a single type it does not recognise makes all of them unreadable -- not just the
    unknown one. libavcodec attaches a type 31 when a JPEG carries an Exif APP1 segment
    and PyAV 18.1 stops at 27, so extract() raised a bare ValueError from inside a
    dependency on almost every photograph ever taken by a camera. Nothing is actually
    lost here: mjpeg exports no motion, no partitions and no QP, so there was never a
    side-data entry this library wanted.
    """
    frames = extract_video(exif_jpeg, pixels=False)
    assert len(frames) == 1
    frame = frames[0]
    assert frame.units, "the fallback grid should still be laid out"
    assert not frame.observed_units
    assert frame.qp_map is None


def test_an_exif_photo_reads_the_same_as_the_same_photo_without_exif(exif_jpeg, still_jpeg):
    """Tolerating the unknown type must not change what is extracted from the picture."""
    plain = extract_video(still_jpeg, pixels=False)[0]
    exif = extract_video(exif_jpeg, pixels=False)[0]
    assert (exif.width, exif.height) == (plain.width, plain.height)
    assert len(exif.units) == len(plain.units)
    assert [(u.x, u.y, u.w, u.h) for u in exif.units] == \
           [(u.x, u.y, u.w, u.h) for u in plain.units]
