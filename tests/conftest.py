import hashlib
import pathlib

import pytest

ASSETS = pathlib.Path(__file__).parent / "assets"


def pytest_sessionstart(session):
    """Fixtures are committed and pinned: a silently different bitstream would change
    every measured number in this suite."""
    sums = (ASSETS / "SHA256SUMS").read_text().split()
    for digest, name in zip(sums[0::2], sums[1::2], strict=True):
        actual = hashlib.sha256((ASSETS / name).read_bytes()).hexdigest()
        assert actual == digest, f"fixture {name} changed; rerun tools/regen_fixtures.sh"


@pytest.fixture(scope="session")
def h264():
    return str(ASSETS / "h264_b3.mp4")


@pytest.fixture(scope="session")
def hevc():
    return str(ASSETS / "hevc_b3.mp4")


@pytest.fixture(scope="session")
def pan():
    return str(ASSETS / "pan.mp4")


@pytest.fixture(scope="session")
def vp9():
    return str(ASSETS / "vp9_aq1.webm")


@pytest.fixture(scope="session")
def vp9_pan():
    """The `pan` shot re-encoded as VP9: same content, one codec with partitions and no
    motion, one with motion and no tree. The only fixture pair that can be compared."""
    return str(ASSETS / "vp9_pan.webm")


@pytest.fixture(scope="session")
def vp9_noaq():
    return str(ASSETS / "vp9_noaq.webm")


@pytest.fixture(scope="session")
def odd():
    return str(ASSETS / "odd_250x170.mp4")


@pytest.fixture(scope="session")
def hd():
    """1080p: padding overshoot (raw MV edge 1088 vs 1080 visible) and the token cap."""
    return str(ASSETS / "hd_1080p.mp4")


@pytest.fixture(scope="session")
def static_box():
    """Near-static scene, one small moving object. 27 of 50 frames have zero motion."""
    return str(ASSETS / "static_box.mp4")


@pytest.fixture(scope="session")
def interlaced():
    """Field-coded, and it DOES export MVs -- the guard is necessary, not theoretical."""
    return str(ASSETS / "interlaced.mp4")


@pytest.fixture(scope="session")
def h264_10bit():
    return str(ASSETS / "h264_10bit.mp4")


@pytest.fixture(scope="session")
def vp8():
    return str(ASSETS / "vp8.webm")


@pytest.fixture(scope="session")
def av1():
    return str(ASSETS / "av1.mp4")


@pytest.fixture(scope="session")
def reschange():
    """Two resolutions concatenated as Annex-B; the decoder re-inits at the second SPS."""
    return str(ASSETS / "reschange.264")
