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
sha256sum *.mp4 > SHA256SUMS
$FF -version | head -1 >  VERSIONS
$FF -hide_banner -h encoder=libx264 2>/dev/null | head -1 >> VERSIONS
# VP9 partition tree. -aq-mode 1 enables segmentation, which is what gates block export;
# the no-aq companion pins the "declared capability != observed capability" case.
$FF -y -v error -f lavfi -i testsrc2=size=640x480:rate=25 -t 2 \
    -c:v libvpx-vp9 -b:v 800k -aq-mode 1 -row-mt 1 -threads 1 -deadline good -cpu-used 4 vp9_aq1.webm
$FF -y -v error -f lavfi -i testsrc2=size=640x480:rate=25 -t 2 \
    -c:v libvpx-vp9 -b:v 800k -threads 1 -deadline good -cpu-used 4 vp9_noaq.webm
sha256sum *.mp4 *.webm > SHA256SUMS
