"""Geometry over coding units: clipping, occupancy, coverage, grid fill.

These are the operations that decide what a "unit" and a "covered pixel" mean, so they
live in one place and both the backend and the patchify layer call the same code. Two
rules are load-bearing:

* **Coverage is a rasterised occupancy mask, never a sum of areas.** ffmpeg emits one
  motion-vector record per (block, prediction list), so a bi-predicted block is reported
  twice with identical geometry. On a reference B-frame, raw record areas sum to 1.70x
  the frame. A mask is idempotent under that duplication; a sum is not.
* **Units live in macroblock-PADDED coded space and must be clipped first.** At 1080p the
  bottom macroblock row reaches y=1087 while the frame is 1080 tall. Clipping before
  rasterising and before normalising keeps 7 rows of padding out of the coordinates.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

from vpatch.types import CodedUnit, UnitKind

# The macroblock side H.264 is defined on, and the cell the `grid` strategy aggregates
# into. Not a tunable dressed as a constant: qp_map() is reported on exactly this grid.
MACROBLOCK = 16


def clip_rect(x: int, y: int, w: int, h: int, width: int, height: int
              ) -> tuple[int, int, int, int] | None:
    """Intersect a coded-space rectangle with the visible frame; None if empty."""
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def rasterise(units: Iterable[CodedUnit], width: int, height: int) -> np.ndarray:
    """Occupancy mask of the area described by `units`."""
    mask = np.zeros((height, width), dtype=bool)
    for u in units:
        mask[u.y:u.y + u.h, u.x:u.x + u.w] = True
    return mask


def coverage(mask: np.ndarray) -> float:
    """Fraction of visible pixels described by observed geometry, in [0, 1]."""
    return float(mask.mean())


def overlap_pixels(units: Sequence[CodedUnit], width: int, height: int) -> int:
    """Pixels claimed by more than one unit.

    Zero for a true tiling (VP9's quadtree). Non-zero for H.264, where a bi-predicted
    block's two records collapse to one unit but neighbouring partitions can still
    double-claim after clipping. Used by tests as an independent oracle.
    """
    counts = np.zeros((height, width), dtype=np.int32)
    for u in units:
        counts[u.y:u.y + u.h, u.x:u.x + u.w] += 1
    return int((counts > 1).sum())


def grid_fill(mask: np.ndarray, width: int, height: int,
              qp_lookup: Callable[[int, int], int | None] | None = None,
              block: int = MACROBLOCK) -> list[CodedUnit]:
    """Cover unobserved area with grid blocks flagged `geometry_observed=False`.

    Holes are regions the codec described by other means -- intra prediction, which is
    100% of an I-frame and typically 10-15% of a P-frame. A hole is not a partition we
    measured, so the rectangle is admitted (a downstream grid needs total coverage) but
    the flag says it is inferred. A consumer that drops `geometry_observed=False` units
    gets exactly what the bitstream said and nothing else.
    """
    fill: list[CodedUnit] = []
    for by in range(0, height, block):
        for bx in range(0, width, block):
            y1, x1 = min(by + block, height), min(bx + block, width)
            if mask[by:y1, bx:x1].all():
                continue
            fill.append(
                CodedUnit(
                    x=bx, y=by, w=x1 - bx, h=y1 - by,
                    kind=UnitKind.INTRA,
                    geometry_observed=False,
                    mvs=(),
                    qp=qp_lookup(bx, by) if qp_lookup else None,
                )
            )
    return fill


def canonical_sort(units: list[CodedUnit]) -> None:
    """Sort in place into the one order extraction is allowed to emit.

    VP9 blocks arrive in tile-decode order and H.264 records in macroblock-thread order,
    both of which vary with `thread_count`. Purity is a contract, so the order is fixed
    here rather than left to whatever the decoder happened to do.
    """
    units.sort(key=lambda u: (u.y, u.x, u.w, u.h))


def grid_shape(width: int, height: int, cell: int = MACROBLOCK) -> tuple[int, int]:
    """(rows, cols) of the aggregation grid covering a visible frame."""
    return (height + cell - 1) // cell, (width + cell - 1) // cell
