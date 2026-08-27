"""Extraction must be a pure function of (file bytes, kwargs).

Not merely a stable token COUNT: identical bytes must give byte-identical values, and
threading must not perturb them. Anything that varies has to travel in kwargs, never in
module or environment state.
"""

import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent("""
    import hashlib, sys
    import numpy as np
    from vpatch.backends import extract_video

    frames = extract_video(sys.argv[1], thread_count=int(sys.argv[2]),
                           fill_holes=sys.argv[3] == "1")
    h = hashlib.sha256()
    for f in frames:
        h.update(f"{f.decode_index}|{f.display_index}|{f.pict_type}|{f.coverage!r}".encode())
        h.update(np.ascontiguousarray(f.qp_map).tobytes() if f.qp_map is not None else b"-")
        h.update(np.ascontiguousarray(f.pixels).tobytes() if f.pixels is not None else b"-")
        for u in f.units:
            h.update(f"{u.x},{u.y},{u.w},{u.h},{u.kind.value if u.kind else "-"},{u.geometry_observed},{u.qp}".encode())
            for m in u.mvs:
                h.update(f"{m.dx!r},{m.dy!r},{m.list_idx}".encode())
    print(h.hexdigest())
""")


def _fingerprint(path, threads, fill=True):
    out = subprocess.run([sys.executable, "-c", SCRIPT, path, str(threads), "1" if fill else "0"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_extraction_is_byte_reproducible_across_processes_and_threads(h264):
    digests = {
        ("p1", 1): _fingerprint(h264, 1),
        ("p2", 1): _fingerprint(h264, 1),
        ("p3", 4): _fingerprint(h264, 4),
        ("p4", 4): _fingerprint(h264, 4),
    }
    assert len(set(digests.values())) == 1, f"extraction is not pure: {digests}"


def test_identical_bytes_give_identical_output(h264, tmp_path):
    copy = tmp_path / "copy.mp4"
    copy.write_bytes(open(h264, "rb").read())
    assert _fingerprint(h264, 1) == _fingerprint(str(copy), 1)


def test_different_bytes_give_different_output(h264, hevc):
    assert _fingerprint(h264, 1) != _fingerprint(hevc, 1)


@pytest.mark.parametrize("fill", [True, False])
def test_vp9_tree_order_is_canonical(vp9, fill):
    """VP9 blocks arrive in tile-decode order, which varies with thread count. Without a
    canonical sort the output differs at thread_count 1 vs 4 -- and it did, until the sort
    was made unconditional rather than a side effect of hole filling.
    """
    assert _fingerprint(vp9, 1, fill) == _fingerprint(vp9, 4, fill)
