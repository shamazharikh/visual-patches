"""The invariants that catch the two silent bugs in MV extraction."""

from collections import Counter

import av
import pytest
from av.codec.context import Flags2

from vpatch.backends import extract_video
from vpatch.backends.ffmpeg_video import max_units_per_frame


def raw_records(path):
    with av.open(path) as c:
        st = c.streams.video[0]
        st.codec_context.flags2 |= Flags2.export_mvs
        out = []
        for f in c.decode(st):
            mv = f.side_data.get("MOTION_VECTORS")
            if mv is not None:
                out.append(mv.to_ndarray())
    return out


def test_dst_is_block_centre_not_top_left(h264):
    """ffmpeg reports the block CENTRE. Exact, zero-tolerance, fixture-independent.

    This replaces the warp-PSNR gate, which passed on this exact bug.
    """
    ok = bad = 0
    for a in raw_records(h264):
        x0 = a["dst_x"].astype(int) - a["w"].astype(int) // 2
        y0 = a["dst_y"].astype(int) - a["h"].astype(int) // 2
        ok += int(((x0 % a["w"] == 0) & (y0 % a["h"] == 0)).sum())
        bad += int(((a["dst_x"] % a["w"] == 0) & (a["dst_y"] % a["h"] == 0)).sum())
    assert ok > 0 and bad == 0, "centre->top-left conversion is not being applied"


def test_block_shapes_are_the_only_four(h264):
    """ffmpeg's add_mb() hard-codes w,h in {8,16}. Structural, not exact counts."""
    shapes = set()
    for a in raw_records(h264):
        shapes |= set(zip(a["w"].tolist(), a["h"].tolist()))
    assert shapes <= {(16, 16), (16, 8), (8, 16), (8, 8)}
    assert len(shapes) >= 2


def test_bipred_records_collapse_to_units(h264):
    """One record per (block, list): a B-frame emits ~1.75x more records than blocks."""
    frames = extract_video(h264, fill_holes=False)
    recs = {len(a) for a in raw_records(h264)}
    b = next(f for f in frames if f.pict_type == "B")
    assert len(b.units) < max(recs)
    for u in b.units:
        assert len(u.mvs) in (1, 2)
        if len(u.mvs) == 2:
            assert {m.list_idx for m in u.mvs} == {0, 1}


def test_coverage_is_a_mask_not_an_area_sum(h264):
    """Summing record areas gives 1.70 on a B-frame. Coverage must stay in [0,1]."""
    for f in extract_video(h264):
        assert 0.0 <= f.coverage <= 1.0
        if f.pict_type == "I":
            assert f.coverage == 0.0, "an I-frame carries no motion vectors"
    b = next(f for f in extract_video(h264, fill_holes=False) if f.pict_type == "B")
    area_sum = sum(u.area for u in b.units) / (b.width * b.height)
    assert area_sum <= 1.0 + 1e-9


def test_holes_are_flagged_not_disguised(h264):
    f = next(f for f in extract_video(h264) if f.pict_type == "I")
    assert f.units and all(not u.geometry_observed for u in f.units)
    covered = sum(u.area for u in f.units)
    assert covered == f.width * f.height


def test_padding_overshoot_is_clipped(odd):
    """250x170 pads to 256x176; unclipped units would index past the frame."""
    for f in extract_video(odd):
        assert f.width == 250 and f.height == 170
        for u in f.units:
            assert u.x >= 0 and u.y >= 0
            assert u.x + u.w <= f.width and u.y + u.h <= f.height


def test_token_cap_is_never_exceeded(h264, odd):
    for path in (h264, odd):
        for f in extract_video(path):
            assert len(f.units) <= max_units_per_frame(f.width, f.height)


def test_known_displacement_pan(pan):
    """bf=0/refs=1 makes the reference knowable, so motion is assertable exactly.

    The crop window advances +4px/frame, so content moves LEFT in the frame and the
    vector -- which points from the block to its source in the reference -- is +4.
    Measured: (4, 0) is 70.6% of all vectors and dy==0 is 81.2%.
    """
    frames = extract_video(pan, fill_holes=False)
    mvs = [(round(m.dx), round(m.dy)) for f in frames for u in f.units for m in u.mvs]
    modal, count = Counter(mvs).most_common(1)[0]
    assert modal == (4, 0), f"expected modal MV (4, 0) for a +4px/frame pan, got {modal}"
    assert count / len(mvs) > 0.5
