#!/bin/bash
WIDTH=1280
HEIGHT=720

# WIDTH=960
# HEIGHT=540

# WIDTH=854
# HEIGHT=480

# WIDTH=640
# HEIGHT=360
QP=23
# VIDEO=tv_show
VIDEO=london

INPUT=/data/zxxia/videos/${VIDEO}/${VIDEO}.mp4
OUTPUT=/data/zxxia/videos/${VIDEO}/${VIDEO}_${WIDTH}x${HEIGHT}_${QP}.mp4

python videos/resize.py \
    --input_video ${INPUT} \
    --output_video ${OUTPUT} \
    --target_width ${WIDTH} \
    --target_height ${HEIGHT} \
    --qp ${QP}
