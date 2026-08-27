"""Gates on the shape contract, and on vpatch staying free of its consumer."""

import subprocess
import sys

import numpy as np
import pytest

from vpatch.backends.ffmpeg_video import extract_video
from vpatch.export import check_projection, grid_thw
from vpatch.patchify import patchify_grid
from vpatch.sampling import anchor_delta, keyframe_anchors

MERGE = 2


@pytest.mark.parametrize("cell", [16, 28, 32])
def test_grid_thw_reproduces_the_token_count_exactly(h264, cell):
    """vLLM checks num_features == prod(grid_thw) // merge**2 and rejects a mismatch.

    Reporting the grid pre-merge makes the division exact for any cell size, rather than
    right only when the cell happens to divide evenly.
    """
    bundle = patchify_grid(extract_video(h264, pixels=False), cell=cell)
    thw = grid_thw(bundle, spatial_merge_size=MERGE)
    assert thw.shape == (1, 3)
    assert int(thw.prod()) // (MERGE * MERGE) == len(bundle.features)


@pytest.mark.parametrize("fixture", ["odd", "hd"])
def test_grid_thw_holds_for_frames_that_are_not_multiples_of_the_cell(request, fixture):
    frames = extract_video(request.getfixturevalue(fixture), pixels=False)
    bundle = patchify_grid(frames, cell=28)
    thw = grid_thw(bundle, spatial_merge_size=MERGE)
    assert int(thw.prod()) // (MERGE * MERGE) == len(bundle.features)
    t, h, w = thw[0]
    assert t == bundle.meta["n_frames"]
    assert (h, w) == tuple(d * MERGE for d in bundle.meta["grid"])


def test_a_pruned_bundle_is_refused_rather_than_given_a_wrong_grid(h264):
    """Pruning removes tokens without removing grid positions, so no dense grid fits."""
    frames = extract_video(h264, pixels=False)
    bundle = patchify_grid(frames)
    pruned, report = anchor_delta(bundle, anchors=keyframe_anchors(frames))
    assert report.dropped > 0
    with pytest.raises(ValueError, match="pruned"):
        grid_thw(pruned)


def test_check_projection_rejects_raw_codec_channels(h264):
    """The failure belongs at this boundary, not inside someone else's forward pass."""
    bundle = patchify_grid(extract_video(h264, pixels=False))
    with pytest.raises(ValueError, match="trained projection"):
        check_projection(bundle.features, hidden_size=3584)
    # A correctly projected array passes.
    check_projection(np.zeros((len(bundle.features), 3584), dtype=np.float32), 3584)
    with pytest.raises(ValueError, match=r"\[N, D\]"):
        check_projection(np.zeros(10, dtype=np.float32), 3584)


def test_vpatch_never_imports_vllm():
    """The integration is documentation; a stray import would make it a dependency."""
    probe = (
        "import sys, importlib;"
        "[importlib.import_module(m) for m in ("
        "'vpatch.types','vpatch.partition','vpatch.patchify','vpatch.sampling',"
        "'vpatch.export','vpatch.cli','vpatch.backends.ffmpeg_video')];"
        "leaked=[m for m in sys.modules if m=='vllm' or m.startswith('vllm.')];"
        "print(','.join(leaked))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "", f"vpatch imported vllm: {out.stdout}"


def test_visualizer_produces_a_self_contained_page(h264, tmp_path):
    """The overlays are how the claims get checked by eye, so they must not break quietly."""
    import re
    import sys

    sys.path.insert(0, "tools")
    from visualize import build

    out = tmp_path / "panels.html"
    build(h264, str(out), frame_index=None, cell=16)
    doc = out.read_text()
    # Self-contained: the picture is embedded, nothing is fetched.
    assert "data:image/jpeg;base64," in doc
    assert "http://" not in doc.replace("http://www.w3.org/2000/svg", "")
    assert doc.count("<figure>") == 7
    # Overlays actually drew something.
    assert doc.count("<rect") > 100
    assert "<line" in doc
    embedded = re.search(r"data:image/jpeg;base64,([A-Za-z0-9+/=]+)", doc).group(1)
    import base64
    assert base64.b64decode(embedded)[:3] == b"\xff\xd8\xff"  # JPEG magic
