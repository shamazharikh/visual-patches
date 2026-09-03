"""Core data model.

Every field here is something the public ffmpeg API actually exports. Fields that a
codec cannot supply are `None` or carry an explicit `*_available` flag in
`PatchBundle.meta` -- we never fabricate a value to fill a column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum

import numpy as np

# ffmpeg's AVMotionVector.motion_scale is a fixed-point divisor (4 = quarter-pel).
# We divide once, at extraction, so nothing downstream sees fixed-point units.


class UnitKind(Enum):
    """What we could observe about a coding unit.

    There is deliberately no SKIP: ffmpeg exports no skip flag. ``AVMotionVector.flags``
    is documented "Currently unused" and is always 0, and skip/direct macroblocks emit
    ordinary MV records carrying their predicted motion. A SKIP member would be an
    inference wearing the costume of an observation.
    """

    INTRA = 0  # no MV records cover this area
    INTER = 1  # at least one MV record

    # Some codecs export geometry without any intra/inter signal at all -- VP9's
    # AVVideoBlockParams carries only (src_x, src_y, w, h, delta_qp). Those units get
    # kind=None rather than a guessed label.


class Modality(StrEnum):
    VIDEO = "video.v1"
    IMAGE = "image.v1"


@dataclass(slots=True, frozen=True)
class MotionVector:
    """One prediction of one block, from one reference list.

    `list_idx` is a prediction-LIST index, not a temporal direction. In low-delay B and
    B-pyramid configurations both L0 and L1 can point into the past. libavutil's own
    header flags the gap: "XXX: set exact relative ref frame reference instead of a
    +/- 1 direction." No reference identity (POC or index) is exported by ffmpeg, so
    MVs cannot be normalized by temporal distance -- see `ref_identity_available`.
    """

    dx: float  # luma pixels
    dy: float  # luma pixels
    list_idx: int  # 0 = L0, 1 = L1

    @property
    def magnitude(self) -> float:
        return float(np.hypot(self.dx, self.dy))


@dataclass(slots=True)
class CodedUnit:
    """One coding block in luma pixel coordinates, top-left anchored.

    ffmpeg reports MV block positions as the block CENTER; the backend converts to
    top-left on the way in. `w`/`h` come from ffmpeg's exporter, which hard-codes
    w,h in {8,16} -- so shapes are drawn from {16x16, 16x8, 8x16, 8x8} and nothing
    finer. Sub-8x8 partitions are collapsed by ffmpeg into a single 8x8 record whose
    motion is one sampled sub-block's; the other three are unrecoverable.
    """

    x: int
    y: int
    w: int
    h: int
    kind: UnitKind | None  # None => the codec exports no intra/inter signal
    geometry_observed: bool  # False => this rectangle is grid fill, not a real partition
    mvs: tuple[MotionVector, ...] = ()
    qp: int | None = None

    @property
    def log2_w(self) -> int:
        # log2 of the side, not a quadtree depth: 16x8 and 8x16 are binary macroblock
        # partitions and have no well-defined quadtree depth.
        return int(self.w).bit_length() - 1

    @property
    def log2_h(self) -> int:
        return int(self.h).bit_length() - 1

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass(slots=True)
class CodedFrame:
    decode_index: int
    display_index: int
    pict_type: str  # 'I' | 'P' | 'B' (and rarer types passed through verbatim)
    key_frame: bool
    width: int
    height: int
    bit_depth: int
    pix_fmt: str
    units: list[CodedUnit] = field(default_factory=list)
    coverage: float = 0.0  # from a rasterised occupancy mask, never a sum of areas
    qp_map: np.ndarray | None = None  # (mb_h, mb_w) int32, absolute QP per macroblock
    concealed: bool = False
    # True when this frame's position in decode order could not be established: PTS is
    # not unique in real streams (variable-frame-rate cameras restamp, and clips in the
    # wild open with two packets sharing a timestamp), so `decode_index` is a best effort
    # and this flag says when to distrust it.
    order_ambiguous: bool = False
    pixels: np.ndarray | None = None  # luma plane, (h, w)

    @property
    def observed_units(self) -> list[CodedUnit]:
        return [u for u in self.units if u.geometry_observed]


@dataclass(slots=True)
class PatchBundle:
    """Tokens handed to a downstream consumer.

    `seq_lens` is required, not optional: variable-length packing needs sample
    boundaries to build a block-diagonal attention mask, and concatenated features
    alone cannot recover them.
    """

    features: np.ndarray  # [N, D] float32
    coords: np.ndarray  # [N, 4] float32 -- cx, cy, w, h normalized to the VISIBLE frame
    times: np.ndarray  # [N] int32 -- per-sample-local display index
    kinds: np.ndarray  # [N] int8
    seq_lens: list[int]
    modality: str
    meta: dict

    def __post_init__(self) -> None:
        n = len(self.features)
        if not (len(self.coords) == len(self.times) == len(self.kinds) == n):
            raise ValueError("PatchBundle arrays must agree in length")
        if sum(self.seq_lens) != n:
            raise ValueError(f"seq_lens sums to {sum(self.seq_lens)}, expected {n}")

    @property
    def cu_seqlens(self) -> np.ndarray:
        """Cumulative sequence boundaries, the form attention kernels want."""
        return np.concatenate([[0], np.cumsum(self.seq_lens)]).astype(np.int32)


class UnsupportedCodecFeature(Exception):
    """Input uses a coding feature whose exported metadata would be silently wrong."""


class CorruptBitstream(Exception):
    """Decoder concealed errors; extracted metadata would be fabricated."""


class ResourceLimitExceeded(Exception):
    """A caller-set bound on decode work was hit.

    Distinct from UnsupportedCodecFeature: nothing about the input is wrong or
    unrepresentable, it is simply larger or slower than this caller allowed.
    """


class ResolutionChanged(Exception):
    """Coded dimensions changed mid-stream; one normalized coordinate space is invalid."""
