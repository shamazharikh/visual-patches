#!/usr/bin/env bash
# Maintainer-only. Fixtures are COMMITTED, not generated at test time: libx264 output
# depends on host thread count, so generating per-run makes every numeric assertion flake
# on a machine with a different core count. -threads 1 pins it.
set -euo pipefail
cd "$(dirname "$0")/../tests/assets"
FF=${FFMPEG:-ffmpeg}
$FF -y -v error -f lavfi -i testsrc2=size=320x240:rate=25 -t 2 \
    -c:v libx264 -bf 3 -g 12 -threads 1 h264_b3.mp4
$FF -y -v error -f lavfi -i testsrc2=size=320x240:rate=25 -t 2 \
    -c:v libx265 -bf 3 -g 12 -threads 1 hevc_b3.mp4
# Known-displacement pan: bf=0/refs=1 makes the reference knowable, so motion magnitude
# and sign are assertable by exact equality instead of a PSNR threshold.
$FF -y -v error -f lavfi -i testsrc2=size=640x480:rate=25 -t 2 \
    -vf "crop=320:240:x='4*n':y=0" -c:v libx264 -bf 0 -refs 1 -g 40 -qp 18 -threads 1 pan.mp4
# Non-multiple-of-16 size: exercises macroblock padding overshoot and clipping.
$FF -y -v error -f lavfi -i testsrc2=size=250x170:rate=25 -t 1 \
    -c:v libx264 -bf 2 -g 12 -threads 1 odd_250x170.mp4
# 1080p: macroblock padding overshoot is only visible at a height that is not a multiple
# of 16. 1080 = 67.5 macroblock rows, so raw MV records reach an exclusive bottom edge of
# 1088 while the API reports 1080. Also the size the hard token cap is asserted at.
$FF -y -v error -f lavfi -i testsrc2=size=1920x1080:rate=25 -t 0.2 \
    -c:v libx264 -bf 2 -g 8 -qp 40 -threads 1 hd_1080p.mp4
# Near-static scene with one small moving object -- the case that breaks naive pruning.
# 27 of its 50 frames contain no moving cell at all, so a "drop zero-motion cells" policy
# deletes them entirely. Generated from a held still frame so the background is genuinely
# static rather than merely low-motion.
$FF -y -v error -f lavfi -i testsrc2=size=320x240:rate=25 -frames:v 1 still.png
$FF -y -v error -loop 1 -i still.png -f lavfi -i "color=c=white:s=16x16:r=25" \
    -filter_complex "[0:v]fps=25[bg];[bg][1:v]overlay=x='40+2*n':y=112:shortest=0" \
    -t 2 -c:v libx264 -bf 2 -g 12 -threads 1 -pix_fmt yuv420p static_box.mp4
rm -f still.png
sha256sum *.mp4 > SHA256SUMS
$FF -version | head -1 >  VERSIONS
$FF -hide_banner -h encoder=libx264 2>/dev/null | head -1 >> VERSIONS
# VP9 partition tree. -aq-mode 1 enables segmentation, which is what gates block export;
# the no-aq companion pins the "declared capability != observed capability" case.
$FF -y -v error -f lavfi -i testsrc2=size=640x480:rate=25 -t 2 \
    -c:v libvpx-vp9 -b:v 800k -aq-mode 1 -row-mt 1 -threads 1 -deadline good -cpu-used 4 vp9_aq1.webm
$FF -y -v error -f lavfi -i testsrc2=size=640x480:rate=25 -t 2 \
    -c:v libvpx-vp9 -b:v 800k -threads 1 -deadline good -cpu-used 4 vp9_noaq.webm
# --- degraded tier and typed-rejection fixtures ---------------------------------
# Interlaced. ffmpeg's field-to-frame MV correction (my *= 2) lives only in the
# IS_16X8/IS_8X16 branches, so 16x16 and 8x8 macroblocks would carry half-magnitude
# vertical motion. This clip DOES export MVs, which is what makes the guard necessary
# rather than theoretical.
$FF -y -v error -f lavfi -i testsrc2=size=320x240:rate=25 -t 1 \
    -c:v libx264 -flags +ilme+ildct -top 1 -bf 0 -threads 1 interlaced.mp4
# 10-bit: QP range shifts by 6*(bit_depth-8), so qp_map values are not comparable
# across bit depths.
$FF -y -v error -f lavfi -i testsrc2=size=320x240:rate=25 -t 1 \
    -pix_fmt yuv420p10le -c:v libx264 -bf 2 -threads 1 h264_10bit.mp4
# VP8 and AV1: the rest of the degraded tier. Neither exports motion or partitions.
$FF -y -v error -f lavfi -i testsrc2=size=320x240:rate=25 -t 1 \
    -c:v libvpx -b:v 400k -threads 1 vp8.webm
$FF -y -v error -f lavfi -i testsrc2=size=320x240:rate=25 -t 0.8 \
    -c:v libaom-av1 -cpu-used 8 -b:v 300k -threads 1 av1.mp4
# Mid-stream resolution change. Annex-B elementary streams concatenate and the decoder
# re-inits at the second SPS; MP4 cannot carry this, so the fixture is a raw .264.
$FF -y -v error -f lavfi -i testsrc2=size=320x240:rate=25 -t 0.4 \
    -c:v libx264 -bf 0 -g 5 -threads 1 -f h264 _a.264
$FF -y -v error -f lavfi -i testsrc2=size=160x120:rate=25 -t 0.4 \
    -c:v libx264 -bf 0 -g 5 -threads 1 -f h264 _b.264
cat _a.264 _b.264 > reschange.264
rm -f _a.264 _b.264
sha256sum *.mp4 *.webm *.264 > SHA256SUMS
