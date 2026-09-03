"""Video extraction through PyAV's public libavcodec API.

Only H.264 exports the metadata this library is built on. The sole producer of
AV_FRAME_DATA_MOTION_VECTORS is ``ff_print_debug_info2()`` in libavcodec/mpegutils.c,
and its only call sites are h264dec.c and mpegvideo_dec.c -- so HEVC, AV1, VP9 and VP8
yield frame types and ordering but no motion and no partitions, at every ffmpeg version.
Those codecs are handled here too, in a degraded tier that reports what it lacks rather
than emitting zeros that look like measurements.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, replace

import av
import numpy as np
from av.codec.context import Flags2
from av.video.frame import PictureType

from vpatch.partition import canonical_sort, clip_rect, grid_fill
from vpatch.types import (
    CodedFrame,
    CodedUnit,
    CorruptBitstream,
    MotionVector,
    ResolutionChanged,
    ResourceLimitExceeded,
    UnitKind,
    UnsupportedCodecFeature,
)

# Codecs whose decoders call ff_print_debug_info2(). Everything else has no motion.
MV_CAPABLE = frozenset({"h264", "mpeg4", "mpeg2video", "mpeg1video", "h263", "vc1", "wmv3"})
# Decoders that attach AVVideoEncParams. These three axes are INDEPENDENT: VP9 exports a
# partition tree and per-block QP but no motion at all, so partitions must not be derived
# from motion-vector availability.
QP_CAPABLE = frozenset({"h264", "mpeg2video", "mpeg1video", "vp9"})
# Codecs whose AVVideoBlockParams describe a genuine non-uniform partition tree. H.264 also
# exports enc-params blocks, but they are a fixed 16x16 macroblock grid, so its geometry
# comes from MV records instead.
TREE_CAPABLE = frozenset({"vp9"})

_PICT_TYPE = {int(m): m.name for m in PictureType}

# ffmpeg's exporter hard-codes w,h in {8,16}; anything else means our assumptions broke.
_VALID_SHAPES = frozenset({(16, 16), (16, 8), (8, 16), (8, 8)})


@dataclass(frozen=True)
class Capability:
    """What a codec CAN export, and -- after probing -- what this file actually does.

    The two differ. VP9 only emits its partition tree when the bitstream uses segmentation:
    an `-aq-mode 1` encode yields blocks on every frame, a default-CRF encode yields none at
    all. A capability answer read off the codec name would call both "partitions=True", so
    `observed` is filled in by trial-decoding real frames.
    """

    codec: str
    motion_vectors: bool
    partitions: bool
    per_block_qp: bool
    ordering: bool = True
    observed: bool | None = None  # None => not probed

    @property
    def degraded(self) -> bool:
        """No geometry of any kind -- neither motion nor a partition tree."""
        return not (self.motion_vectors or self.partitions)


def canonical_codec(codec_context) -> str:
    """The codec's own name, not the name of whichever decoder ffmpeg picked.

    `codec_context.name` is the DECODER: an AV1 stream reads back as "libdav1d", and a
    build linked against libopenh264 would report "libopenh264" for ordinary H.264. Since
    the capability table is keyed by codec, reading the decoder name would have quietly
    demoted a fully capable file to the degraded tier -- a false "no motion available"
    on input this library handles, which is the worst direction for the error to run.
    """
    codec = codec_context.codec
    return getattr(codec, "canonical_name", None) or codec_context.name


def capability_for(codec: str) -> Capability:
    return Capability(
        codec=codec,
        motion_vectors=codec in MV_CAPABLE,
        partitions=codec in MV_CAPABLE or codec in TREE_CAPABLE,
        per_block_qp=codec in QP_CAPABLE,
    )


def open_container(path: str):
    """Open a container, converting libav's errors into this library's typed ones.

    A truncated MP4 is the common real-world case and it fails here rather than during
    decode: the moov atom lives at the end of the file, so losing the tail loses the
    index and the container will not open at all. Letting PyAV's InvalidDataError escape
    would make callers catch a dependency's exception type to handle a condition this
    library has a name for.
    """
    try:
        return av.open(path)
    except av.error.InvalidDataError as exc:
        raise CorruptBitstream(f"cannot parse container {path!r}: {exc}") from exc
    except av.error.FFmpegError as exc:
        raise CorruptBitstream(f"cannot open {path!r}: {exc}") from exc


def _packet_order(path: str) -> tuple[dict[int, list[int]], int]:
    """Map presentation timestamp -> decode positions, and count the video packets.

    Decode order is the packet order by DTS; display order is by PTS. Frame-level
    ``pkt_dts`` is reported post-reordering and will not show the divergence, so we read
    it from the packet stream instead.

    PTS is **not unique** in real streams. A surveillance clip in this corpus opens with
    two packets both stamped pts=77, and variable-frame-rate camera output does this
    routinely. So the map is pts -> list of decode ranks, consumed in order, and the
    packet count is returned separately: deriving it from the size of a pts-keyed dict
    undercounts by exactly the number of duplicates, which turned a healthy 84-frame clip
    into a spurious "decoder was not drained" failure.
    """
    with open_container(path) as container:
        stream = container.streams.video[0]
        packets = [(p.pts, p.dts) for p in container.demux(stream) if p.size]
    return rank_packets(packets), len(packets)


def rank_packets(packets: list[tuple[int | None, int | None]]) -> dict[int, list[int]]:
    """pts -> decode ranks, ascending. Pure, so the duplicate cases are testable."""
    # Index-tiebroken so equal DTS resolves to demux order rather than sort internals.
    order = sorted(range(len(packets)),
                   key=lambda i: (packets[i][1] is None, packets[i][1], i))
    rank_of = [0] * len(packets)
    for rank, i in enumerate(order):
        rank_of[i] = rank

    ranks: dict[int, list[int]] = {}
    for i, (pts, _) in enumerate(packets):
        if pts is not None:
            ranks.setdefault(pts, []).append(rank_of[i])
    for v in ranks.values():
        v.sort()
    return ranks


def _units_from_mvs(arr: np.ndarray, width: int, height: int, qp_map: np.ndarray | None
                    ) -> tuple[list[CodedUnit], np.ndarray]:
    """Group raw AVMotionVector records into coding units, and rasterise occupancy.

    ffmpeg emits one record per (block, prediction list), so a bi-predicted block appears
    twice with identical geometry. Summing record areas therefore overshoots the frame --
    measured 1.70x on a B-frame -- which is why occupancy is rasterised into a mask and
    coverage is taken from the mask, never from an area sum.
    """
    # dst_x/dst_y are the block CENTRE. Without this conversion, 0 of 612 blocks on a
    # reference B-frame land on their own grid; with it, all 612 do.
    x0 = arr["dst_x"].astype(np.int32) - arr["w"].astype(np.int32) // 2
    y0 = arr["dst_y"].astype(np.int32) - arr["h"].astype(np.int32) // 2

    scale = arr["motion_scale"].astype(np.float32)
    scale[scale == 0] = 1.0
    dx = arr["motion_x"].astype(np.float32) / scale
    dy = arr["motion_y"].astype(np.float32) / scale
    # source is a prediction-LIST index: -1 => L0, +1 => L1. It is not a time direction.
    list_idx = (arr["source"].astype(np.int32) > 0).astype(np.int8)

    grouped: dict[tuple[int, int, int, int], list[MotionVector]] = {}
    for i in range(len(arr)):
        key = (int(x0[i]), int(y0[i]), int(arr["w"][i]), int(arr["h"][i]))
        grouped.setdefault(key, []).append(
            MotionVector(dx=float(dx[i]), dy=float(dy[i]), list_idx=int(list_idx[i]))
        )

    occupancy = np.zeros((height, width), dtype=bool)
    units: list[CodedUnit] = []
    for (ux, uy, uw, uh), mvs in grouped.items():
        # Coordinates live in macroblock-PADDED coded space; the bottom/right rows
        # overhang the visible frame and must be clipped before anything is normalized.
        rect = clip_rect(ux, uy, uw, uh, width, height)
        if rect is None:
            continue
        cx0, cy0, cw, ch = rect
        occupancy[cy0:cy0 + ch, cx0:cx0 + cw] = True
        units.append(
            CodedUnit(
                x=cx0, y=cy0, w=cw, h=ch,
                kind=UnitKind.INTER,
                geometry_observed=True,
                mvs=tuple(sorted(mvs, key=lambda m: m.list_idx)),
                qp=_qp_at(qp_map, cx0, cy0),
            )
        )
    canonical_sort(units)
    return units, occupancy


def _units_from_enc_params(enc, width: int, height: int
                           ) -> tuple[list[CodedUnit], np.ndarray]:
    """Build units from AVVideoBlockParams -- VP9's superblock quadtree.

    Unlike H.264 MV records, this is an exact tiling: every pixel is covered exactly once,
    on every frame including I-frames (measured coverage 1.0000, 0 overlapping pixels).
    Depths run 64/32/16/8. `kind` is None because AVVideoBlockParams carries no intra/inter
    signal -- only (src_x, src_y, w, h, delta_qp).

    Blocks are emitted in tile-decode order, which varies with thread count, so the caller's
    canonical (y, x) sort is what keeps extraction reproducible.
    """
    occupancy = np.zeros((height, width), dtype=bool)
    units: list[CodedUnit] = []
    base_qp = int(enc.qp)
    for i in range(enc.nb_blocks):
        b = enc.block_params(i)
        rect = clip_rect(int(b.src_x), int(b.src_y), int(b.w), int(b.h), width, height)
        if rect is None:
            continue
        x0, y0, bw, bh = rect
        occupancy[y0:y0 + bh, x0:x0 + bw] = True
        units.append(
            CodedUnit(
                x=x0, y=y0, w=bw, h=bh,
                kind=None,
                geometry_observed=True,
                mvs=(),
                qp=base_qp + int(b.delta_qp),
            )
        )
    # Blocks arrive in tile-decode order, which varies with thread count. The tiling is
    # exact, so (y, x) is a unique canonical key -- without this sort, extraction is not
    # reproducible across thread counts.
    canonical_sort(units)
    return units, occupancy


def _qp_at(qp_map: np.ndarray | None, x: int, y: int) -> int | None:
    if qp_map is None:
        return None
    mb_y, mb_x = y // 16, x // 16
    if 0 <= mb_y < qp_map.shape[0] and 0 <= mb_x < qp_map.shape[1]:
        return int(qp_map[mb_y, mb_x])
    return None


class VideoExtractor:
    def __init__(self, path: str, *, pixels: bool = True, strict: bool = True,
                 fill_holes: bool = True, max_frames: int | None = None,
                 max_pixels: int = 8192 * 8192, max_decode_seconds: float | None = None,
                 thread_count: int = 1):
        self.path = path
        self.pixels = pixels
        self.strict = strict
        self.fill_holes = fill_holes
        self.max_frames = max_frames
        self.max_pixels = max_pixels
        # Wall-clock bound on the whole read, demux included: `_packet_order` walks every
        # packet in the file before decoding starts, so a cap that only covered the decode
        # loop would not bound the work on a large input.
        self.max_decode_seconds = max_decode_seconds
        # Threading is part of the purity contract: it must not change the output.
        self.thread_count = thread_count

    def capability(self, *, probe_frames: int = 0) -> Capability:
        """Declared capability; with probe_frames > 0, trial-decode to confirm it."""
        with open_container(self.path) as container:
            cap = capability_for(
                canonical_codec(container.streams.video[0].codec_context)
            )
        if not probe_frames:
            return cap
        found = False
        for frame in VideoExtractor(self.path, pixels=False, fill_holes=False,
                                    max_frames=probe_frames).extract():
            if frame.observed_units:
                found = True
                break
        return replace(cap, observed=found)

    def extract(self) -> list[CodedFrame]:
        deadline = (time.monotonic() + self.max_decode_seconds
                    if self.max_decode_seconds is not None else None)
        decode_ranks, n_packets = _packet_order(self.path)
        self._check_deadline(deadline, 0)
        # Frames arrive from the decoder in display order; each one claims the next
        # unused decode rank for its PTS, so duplicate timestamps stay distinguishable.
        claimed: Counter[int] = Counter()

        with open_container(self.path) as container:
            stream = container.streams.video[0]
            ctx = stream.codec_context
            codec = canonical_codec(ctx)
            cap = capability_for(codec)
            ctx.thread_count = self.thread_count
            if cap.motion_vectors:
                ctx.flags2 |= Flags2.export_mvs
            if cap.motion_vectors or cap.partitions or cap.per_block_qp:
                ctx.options = {"export_side_data": "mvs+venc_params"}

            frames: list[CodedFrame] = []
            first_dims: tuple[int, int] | None = None

            # `container.decode()` drains the decoder at end of stream. A hand-rolled
            # demux/decode loop without a flush silently drops trailing B-frames -- 48 of
            # 50 on the reference fixture -- and still yields a valid index permutation.
            for frame in container.decode(stream):
                if frame.width * frame.height > self.max_pixels:
                    raise ResourceLimitExceeded(
                        f"frame {frame.width}x{frame.height} exceeds "
                        f"max_pixels={self.max_pixels}"
                    )
                self._check_deadline(deadline, len(frames))
                if first_dims is None:
                    first_dims = (frame.width, frame.height)
                elif (frame.width, frame.height) != first_dims:
                    raise ResolutionChanged(
                        f"coded size changed {first_dims} -> {(frame.width, frame.height)}"
                    )
                if getattr(frame, "interlaced_frame", False):
                    raise UnsupportedCodecFeature(
                        "interlaced/field-coded input: ffmpeg's field-to-frame MV correction "
                        "(my *= 2) exists only in the IS_16X8/IS_8X16 branches, so 16x16 and "
                        "8x8 macroblocks would carry half-magnitude vertical motion"
                    )
                frames.append(self._build(frame, cap, decode_ranks, claimed))
                if self.max_frames and len(frames) >= self.max_frames:
                    break

        if self.max_frames is None and len(frames) != n_packets:
            raise CorruptBitstream(
                f"decoded {len(frames)} frames from {n_packets} video packets "
                "(decoder was not drained, or frames were dropped)"
            )

        for display_index, frame in enumerate(sorted(frames, key=lambda f: f.display_index)):
            frame.display_index = display_index
        return frames

    def _check_deadline(self, deadline: float | None, decoded: int) -> None:
        if deadline is not None and time.monotonic() > deadline:
            raise ResourceLimitExceeded(
                f"decode exceeded max_decode_seconds={self.max_decode_seconds} "
                f"after {decoded} frames"
            )

    def _build(self, frame, cap: Capability, decode_ranks: dict[int, list[int]],
               claimed: Counter[int]) -> CodedFrame:
        w, h = frame.width, frame.height
        corrupt = bool(getattr(frame, "is_corrupt", False))
        if corrupt and self.strict:
            raise CorruptBitstream(
                f"decoder flagged frame pts={frame.pts} corrupt; concealed metadata is fabricated"
            )

        # PyAV maps every side-data entry through an IntEnum while building the
        # container, so a single type it does not recognise makes ALL of them
        # unreadable, not just the unknown one. An Exif APP1 segment is enough:
        # libavcodec attaches side data of type 31 for it and PyAV 18.1's enum stops at
        # 27, so `extract()` raised a bare ValueError from inside a dependency on almost
        # any photograph a camera has taken. Isolated by stripping segments one at a
        # time -- APP1 is the trigger, and a 36-byte minimal Exif reproduces it.
        #
        # Nothing is lost when the codec exports no metadata to begin with, which is the
        # still-image case. When it does, the loss is real, and strict mode refuses
        # rather than reporting an unread frame as a measured empty one.
        try:
            side = frame.side_data
        except ValueError as exc:
            exports_something = (cap.motion_vectors or cap.partitions or cap.per_block_qp)
            if exports_something and self.strict:
                raise UnsupportedCodecFeature(
                    f"libav attached a side-data type this build of PyAV cannot "
                    f"enumerate ({exc}); {cap.codec} exports metadata that therefore "
                    f"cannot be read from frame pts={frame.pts}"
                ) from exc
            side = None

        qp_map = None
        enc = side.get("VIDEO_ENC_PARAMS") if side is not None else None
        if enc is not None and cap.codec not in TREE_CAPABLE:
            # qp_map() assumes a fixed macroblock grid; for a variable tree the per-block
            # delta_qp on each unit is the honest representation.
            try:
                qp_map = np.asarray(enc.qp_map(), dtype=np.int32)
            except Exception:
                qp_map = None

        units: list[CodedUnit] = []
        occupancy = np.zeros((h, w), dtype=bool)

        # VP9 geometry comes from the enc-params block tree, not from motion.
        if cap.codec in TREE_CAPABLE and enc is not None and enc.nb_blocks:
            units, occupancy = _units_from_enc_params(enc, w, h)

        mv = (side.get("MOTION_VECTORS")
              if cap.motion_vectors and side is not None else None)
        if mv is not None:
            arr = mv.to_ndarray()
            shapes = set(zip(arr["w"].tolist(), arr["h"].tolist(), strict=True))
            if not shapes <= _VALID_SHAPES:
                raise UnsupportedCodecFeature(
                    f"unexpected MV block shapes: {shapes - _VALID_SHAPES}")
            units, occupancy = _units_from_mvs(arr, w, h, qp_map)

        pict = _PICT_TYPE.get(int(frame.pict_type), str(frame.pict_type))
        # An I-frame carries no motion by construction; suppressing MVs on a concealed
        # I-frame is what stops a bit-flip from manufacturing 23k fake vectors.
        if corrupt and pict == "I":
            units, occupancy = [], np.zeros((h, w), dtype=bool)

        coverage = float(occupancy.mean())
        if self.fill_holes:
            units = units + grid_fill(
                occupancy, w, h, lambda x, y: _qp_at(qp_map, x, y)
            )
        canonical_sort(units)

        # A frame whose PTS matches no packet, or a second frame claiming a PTS that
        # only one packet carried, has no recoverable decode position. That is reported
        # as ambiguous rather than silently handed a -1 that reads like an index.
        ranks = decode_ranks.get(frame.pts) if frame.pts is not None else None
        seq = claimed[frame.pts] if frame.pts is not None else 0
        if ranks is not None and seq < len(ranks):
            decode_index = ranks[seq]
            ambiguous = len(ranks) > 1
        else:
            decode_index = -1
            ambiguous = True
        if frame.pts is not None:
            claimed[frame.pts] += 1

        return CodedFrame(
            decode_index=decode_index,
            order_ambiguous=ambiguous,
            display_index=frame.pts if frame.pts is not None else 0,
            pict_type=pict,
            key_frame=bool(frame.key_frame),
            width=w, height=h,
            bit_depth=frame.format.components[0].bits if frame.format.components else 8,
            pix_fmt=frame.format.name,
            units=units,
            coverage=coverage,
            qp_map=qp_map,
            concealed=corrupt,
            pixels=frame.to_ndarray(format="gray")[:h, :w] if self.pixels else None,
        )


def extract_video(path: str, **kw) -> list[CodedFrame]:
    return VideoExtractor(path, **kw).extract()


def max_units_per_frame(width: int, height: int) -> int:
    """Hard upper bound on distinct coding units in one frame.

    ffmpeg's exporter can only emit w,h in {8,16}, so the finest partition is 8x8 and the
    bound is exact -- no sampling estimate needed for a token budget.
    """
    return 4 * math.ceil(width / 16) * math.ceil(height / 16)
