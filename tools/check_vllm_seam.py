#!/usr/bin/env python3
"""Re-verify every claim in docs/vllm-integration.md against an INSTALLED vLLM.

    uv run --with vllm python tools/check_vllm_seam.py

The doc states version-specific facts about someone else's fast-moving codebase. Rather
than let them rot silently, each one is checked here, and the script exits non-zero when
one stops holding -- so the doc can be corrected instead of believed.

Checks are made by parsing vLLM's source, not by importing it: `import vllm` pulls in
torch and a CUDA stack, and none of that is needed to read a dict literal or a dataclass
field. That keeps this runnable in CI, and on a machine with no GPU.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

RESULTS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def _vllm_root() -> pathlib.Path | None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        return None
    return pathlib.Path(spec.origin).parent


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _dict_literal_keys(tree: ast.Module, name: str) -> set[str] | None:
    """Keys of a module-level `name: ... = {...}` literal."""
    for node in tree.body:
        targets = ([node.target] if isinstance(node, ast.AnnAssign)
                   else getattr(node, "targets", []))
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        if isinstance(node.value, ast.Dict):
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    return None


def _version(root: pathlib.Path) -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("vllm")
    except PackageNotFoundError:
        return "unknown"


def _is_mutated(tree: ast.Module, name: str) -> bool:
    """Any rebinding, `.update(...)`, or subscript ASSIGNMENT of `name` after its literal.

    A subscript READ (`MM_PARSER_MAP[part_type]`) is how the map is used and does not
    count; only a write would make the map extensible.
    """
    assigns = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == name:
            assigns += 1
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    assigns += 1
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) \
                        and t.value.id == name:
                    return True
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == name:
            if node.func.attr in {"update", "setdefault", "pop", "clear"}:
                return True
    return assigns != 1


def main() -> int:
    root = _vllm_root()
    if root is None:
        print("vllm is not installed; run with `uv run --with vllm`", file=sys.stderr)
        return 2
    print(f"vLLM {_version(root)} at {root}\n")

    chat_utils = _parse(root / "entrypoints" / "chat_utils.py")
    keys = _dict_literal_keys(chat_utils, "MM_PARSER_MAP")
    if not check("MM_PARSER_MAP is a module-level dict literal", keys is not None):
        return 1
    assert keys is not None
    check("MM_PARSER_MAP is closed -- assigned once, never mutated",
          not _is_mutated(chat_utils, "MM_PARSER_MAP"),
          f"{len(keys)} keys, so a new modality cannot register one")
    check("chat content parts accept image_embeds and audio_embeds",
          {"image_embeds", "audio_embeds"} <= keys)
    check("chat content parts do NOT accept video_embeds", "video_embeds" not in keys,
          "the OpenAI-compatible server cannot take video embeddings")

    mm_config = (root / "config" / "multimodal.py").read_text()
    check("MultiModalConfig.enable_mm_embeds exists",
          "enable_mm_embeds: bool" in mm_config)

    plugins = (root / "plugins" / "__init__.py").read_text()
    check("plugin entrypoint group is vllm.general_plugins",
          'DEFAULT_PLUGINS_GROUP = "vllm.general_plugins"' in plugins)

    qwen = (root / "model_executor" / "models" / "qwen2_vl.py").read_text()
    check("Qwen2-VL accepts video_embeds + video_grid_thw",
          "class Qwen2VLVideoEmbeddingInputs" in qwen and "video_grid_thw" in qwen)
    check("num_video_features == prod(grid_thw) // spatial_merge_size**2",
          "video_grid_sizes // spatial_merge_size // spatial_merge_size" in qwen,
          "the identity vpatch.export.grid_thw is built to satisfy")

    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} claims hold")
    if failed:
        print("docs/vllm-integration.md is out of date for this vLLM version.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
