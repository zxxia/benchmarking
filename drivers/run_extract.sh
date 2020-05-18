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
# VIDEO=chicago_virtual_run
# VIDEO=tv_show
VIDEO=london

INPUT=/data/zxxia/videos/${VIDEO}/${VIDEO}_${WIDTH}x${HEIGHT}_${QP}.mp4
OUTPUT=/data/zxxia/videos/${VIDEO}/${HEIGHT}p

python videos/extract_frames.py \
    --input_video ${INPUT} \
    --output_image_path ${OUTPUT} \
    --qscale 2
