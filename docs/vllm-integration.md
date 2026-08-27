# Using `vpatch` with vLLM

**Status: documentation, not a plugin.** `vpatch` has no vLLM dependency and never
imports it — there is a test asserting that. This file explains the one seam that
actually works, the ones that do not, and what is still missing to make the working seam
useful.

Every claim below was verified against **vLLM 0.28.0** on 2026-08-27 by reading the
installed source. To re-verify against your own version:

```bash
uv run --with vllm python tools/check_vllm_seam.py
```

It exits non-zero if any claim has stopped holding, so this document can be corrected
rather than believed. It parses vLLM's source instead of importing it, so it needs no GPU
and no model download.

## The short version

There is no supported way to register a *new modality* with vLLM from an out-of-tree
plugin. There **is** a supported way to hand vLLM precomputed embeddings for an existing
modality, and that is the seam to use.

What is missing is not plumbing: it is a **trained projection** from `vpatch`'s ~21 codec
channels into a language backbone's embedding space. This project deliberately ships no
weights, so the integration below is complete up to that adapter and no further.

## What does not work, and why

**A new modality cannot be registered.** vLLM's plugin entrypoint group
(`vllm.general_plugins`) is real, and a plugin declaring one *is* discovered and its
`register()` *is* called — an earlier version of this document claimed the entrypoint
could never be reached, which was wrong. It is reached; it simply has nothing useful to
do. Registering a genuinely new modality would need a `BaseProcessingInfo`, a
`BaseDummyInputsBuilder`, a `BaseMultiModalProcessor`, and a model class implementing
`SupportsMultiModal` — and vLLM's own out-of-tree example only *subclasses an existing
in-tree model*. There is no blessed path for a modality with no model to inherit from, so
the failure surfaces later and less legibly, at `_get_model_cls`.

**The OpenAI-compatible server cannot take video embeddings at all.**
`MM_PARSER_MAP` in `vllm/entrypoints/chat_utils.py` is a module-level dict literal,
assigned once and never mutated, with 15 keys. It has `image_embeds` and `audio_embeds`
but **no `video_embeds`**, and no registration hook. A chat request therefore cannot
carry video embeddings, whatever the model supports.

That leaves the offline `LLM` class, where embeddings travel as tensors under
`multi_modal_data`. `MultiModalConfig.enable_mm_embeds` documents exactly this split:
*"for `LLM` class, this refers to tensor inputs under `multi_modal_data`; for the
OpenAI-compatible server, this refers to chat messages with content `type: "*_embeds"`."*

## What does work

Serve with `--enable-mm-embeds` and pass tensors directly. For Qwen2-VL the video
embedding path takes two arrays:

| field | shape | meaning |
|---|---|---|
| `video_embeds` | `(num_video_features, hidden_size)` | one row per token |
| `video_grid_thw` | `(num_videos, 3)` | `(grid_t, grid_h, grid_w)` |

with the count checked as

```python
num_video_features == prod(video_grid_thw) // spatial_merge_size ** 2
```

`hidden_size` must equal the language backbone's hidden size.

### The grid lines up exactly

`vpatch`'s layout satisfies that identity by construction. A bundle of `T` frames on a
`rows x cols` cell grid has `T * rows * cols` tokens, so reporting the grid *pre-merge* —
`grid_h = rows * merge`, `grid_w = cols * merge` — makes the division exact for any cell
size and any frame count. `vpatch.export.grid_thw` does this and refuses rather than
guesses when it cannot:

```python
from vpatch.backends.ffmpeg_video import extract_video
from vpatch.patchify import patchify_grid
from vpatch.export import grid_thw, check_projection

frames = extract_video("clip.mp4", pixels=False)
bundle = patchify_grid(frames, cell=28)

thw = grid_thw(bundle)                  # e.g. [[50, 18, 24]] for 5400 tokens
check_projection(bundle.features, hidden_size=3584)   # raises: 21 != 3584
```

`check_projection` is a guard, not a projection. It fails loudly at the boundary rather
than letting a shape mismatch surface inside someone else's forward pass.

### Pruned bundles have no grid

`grid_thw` accepts only an unpruned bundle. Anchored pruning removes *tokens* without
removing *grid positions*, so no dense `(t, h, w)` describes the result. A consumer of
pruned tokens needs the per-token `coords` and `times` — and a model that can attend over
a variable-length set, which the dense grid path is not. `seq_lens`/`cu_seqlens` on the
bundle exist for exactly that NaViT-style packing.

## What is still missing

A trained adapter mapping `[N, 21] -> [N, hidden_size]`. `vpatch` emits interpretable
codec channels — motion means and standard deviations per prediction list, validity
fractions, partition geometry, quantiser statistics — and a backbone wants several
thousand dimensions of its own learned space. Feeding raw channels through a random
projection produces confident nonsense, which is why this library ships neither.

Training that adapter is out of scope by decision: `vpatch` is an extraction library. What
it guarantees is that the features are honest — every channel is something the codec
actually exported, absent measurements are flagged rather than zero-filled, and extraction
is a pure function of the file bytes.

## Optional extra, and a version constraint worth knowing

`pyproject.toml` declares an unused `[project.optional-dependencies] vllm` extra so
`pip install vpatch[vllm]` provisions a compatible vLLM. Nothing in the runtime package
imports it.

**vLLM caps numpy.** 0.28.0 pins `numba==0.65.0`, which requires `numpy>=1.22,<2.5`. A
`numpy>=2.5` floor on this package therefore makes `vpatch[vllm]` unsatisfiable — the
resolver fails outright rather than degrading. `vpatch` floors numpy at `1.26` for that
reason and for no other; nothing here needs a recent version. The test suite runs on the
resolved numpy, so it is exercised at the version a vLLM user would actually get.
