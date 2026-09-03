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


def _encode_jpeg(path: str, dest) -> str:
    """Frame 0 of a fixture, written out as a JPEG.

    Built at runtime rather than committed: a single intra frame has no inter-frame
    bitstream behaviour worth pinning by sha256, which is the only reason the other
    fixtures are binaries in the tree.
    """
    import av
    import numpy as np

    from vpatch.backends.ffmpeg_video import VideoExtractor

    luma = VideoExtractor(path, pixels=True, max_frames=1).extract()[0].pixels
    rgb = np.repeat(luma[:, :, None], 3, axis=2)
    with av.open(str(dest), mode="w", format="mjpeg") as container:
        stream = container.add_stream("mjpeg", rate=1)
        stream.width, stream.height = luma.shape[1], luma.shape[0]
        stream.pix_fmt = "yuvj420p"
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(dest)


@pytest.fixture
def still_jpeg(h264, tmp_path):
    return _encode_jpeg(h264, tmp_path / "still.jpg")


@pytest.fixture
def exif_jpeg(still_jpeg, tmp_path):
    """The same still with a minimal Exif APP1 segment -- one IFD0 orientation entry.

    Almost every camera JPEG carries Exif, and libavcodec turns it into a frame
    side-data entry of a type PyAV's enum does not cover, which is enough to make the
    whole side-data container unreadable.
    """
    import struct

    tiff = b"II*\x00" + struct.pack("<I", 8)
    ifd = (struct.pack("<H", 1)
           + struct.pack("<HHI", 0x0112, 3, 1) + struct.pack("<I", 1)
           + struct.pack("<I", 0))
    body = b"Exif\x00\x00" + tiff + ifd
    app1 = b"\xff\xe1" + struct.pack(">H", len(body) + 2) + body

    raw = pathlib.Path(still_jpeg).read_bytes()
    end = 2 + 2 + struct.unpack(">H", raw[4:6])[0]   # APP1 must follow APP0, not precede it
    dest = tmp_path / "exif.jpg"
    dest.write_bytes(raw[:end] + app1 + raw[end:])
    return str(dest)
