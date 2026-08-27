"""Shaping a `PatchBundle` for a consumer that accepts precomputed embeddings.

This module is arithmetic only. It imports nothing from vLLM and never will -- see
`docs/vllm-integration.md` for why the integration is documentation rather than a plugin.

The one thing worth encoding in code is the shape contract, because it is easy to get
subtly wrong and fails deep inside someone else's model. vLLM's Qwen2-VL path accepts
`video_embeds` of shape `(num_video_features, hidden_size)` alongside `video_grid_thw` of
shape `(num_videos, 3)`, and computes the expected feature count as

    num_video_features == prod(video_grid_thw) // spatial_merge_size ** 2

so a grid that does not satisfy that identity is rejected after the request is already
in flight.
"""

from __future__ import annotations

import numpy as np

from vpatch.types import PatchBundle


def grid_thw(bundle: PatchBundle, *, spatial_merge_size: int = 2) -> np.ndarray:
    """A `(1, 3)` `(grid_t, grid_h, grid_w)` consistent with this bundle's token count.

    The spatial merge divides the grid by `spatial_merge_size` on each axis, so the grid
    is reported pre-merge: `grid_h = rows * merge`, `grid_w = cols * merge`. That makes
    the identity exact rather than approximately right, for any cell size and any frame
    count.

    Only valid for an unpruned bundle. Pruning removes tokens without removing grid
    positions, so no dense `(t, h, w)` describes the result -- a consumer of pruned
    tokens needs the per-token `coords`/`times`, not a grid.
    """
    if len(bundle.seq_lens) != 1:
        raise ValueError("grid_thw describes one sample; split the bundle first")
    rows, cols = bundle.meta["grid"]
    t = bundle.meta["n_frames"]
    expected = t * rows * cols
    if len(bundle.features) != expected:
        raise ValueError(
            f"bundle has {len(bundle.features)} tokens but its grid implies {expected}; "
            "this bundle looks pruned, and a pruned bundle has no dense grid"
        )
    m = spatial_merge_size
    thw = np.array([[t, rows * m, cols * m]], dtype=np.int64)
    if int(thw.prod()) // (m * m) != len(bundle.features):
        raise ValueError("grid_thw does not reproduce the token count")
    return thw


def check_projection(features: np.ndarray, hidden_size: int) -> None:
    """Raise unless `features` is already in the backbone's embedding space.

    Deliberately a guard and not a projection. `vpatch` emits ~21 interpretable codec
    channels; a language backbone wants several thousand dimensions of its own learned
    space. Mapping between them is a trained adapter, and this project ships no weights,
    so the honest thing to expose is the check that catches the mistake early rather
    than a random matrix that produces confident nonsense.
    """
    if features.ndim != 2:
        raise ValueError(f"expected [N, D] features, got shape {features.shape}")
    if features.shape[1] != hidden_size:
        raise ValueError(
            f"features are {features.shape[1]}-dimensional but the backbone expects "
            f"{hidden_size}. vpatch emits raw codec channels, not backbone embeddings -- "
            "a trained projection has to sit between them, and this library ships none."
        )
