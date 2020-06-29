# Object Detection Module

## Example Usage(Inference)
```bash
WIDTH=1280
HEIGHT=720
QP=23
VIDEO=tv_show
RESOL=${HEIGHT}p

GPU=0

FASTER_RCNN_RESNET101=/data/zxxia/tensorflow_models/faster_rcnn_resnet101_coco_2018_01_28
FASTER_RCNN_INCEPTION_V2=/data/zxxia/tensorflow_models/faster_rcnn_inception_v2_coco_2018_01_28
SSD_MOBILENET_V2=/data/zxxia/tensorflow_models/ssd_mobilenet_v2_coco_2018_03_29

cd {PathToBenchmarking}/benchmarking
python object_detection \
    --device ${GPU} \
    --model ${FASTER_RCNN_RESNET101} \
    --input_path /data/zxxia/videos/${VIDEO}/${RESOL} \
    --width ${WIDTH} \
    --height ${HEIGHT} \
    --qp ${QP} \
    --output_path /data/zxxia/videos/${VIDEO}/profile
```

## Example Usage(Training)

TODO
