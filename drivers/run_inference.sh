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

VIDEO=london
# VIDEO=chicago_virtual_run
# VIDEO=tv_show
RESOL=${HEIGHT}p

GPU=1

FASTER_RCNN_RESNET101=/data/zxxia/models/research/object_detection/faster_rcnn_resnet101_coco_2018_01_28
SSD_MOBILENET_V2=/data/zxxia/models/research/object_detection/ssd_mobilenet_v2_coco_2018_03_29

# mkdir /data/zxxia/videos/${VIDEO}/profile
    # --model ${SSD_MOBILENET_V2} \
python object_detection \
    --device ${GPU} \
    --model ${FASTER_RCNN_RESNET101} \
    --input_path /data/zxxia/videos/${VIDEO}/${RESOL}_cropped \
    --width ${WIDTH} \
    --height ${HEIGHT} \
    --qp ${QP} \
    --crop \
    --output_path /data/zxxia/videos/${VIDEO}/profile
