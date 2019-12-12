#!/bin/bash

# Change CODE_PATH to the object detection path.
CODE_PATH="/home/zxxia/models/research/object_detection"
# Change full model path to path where trained object detection model is saved.

MODEL_PATH=/data/zxxia/ekya/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb
# Change DATA_PATH to downloaded Youtube video path
# DATA_PATH='/mnt/data/zhujun/dataset/Youtube/'
DATA_PATH='/data/zxxia/ekya/datasets/waymo_images'
# Change dataset_list to your dataset name

CAMERA_LIST="FRONT FRONT_LEFT FRONT_RIGHT SIDE_LEFT SIDE_RIGHT"

GPU=2
RESIZE_RESOL_LIST='720p' # 540p 480p 360p'
QP_LIST="original" #'30 35 40'

SEGMENTS=$(ls -d ${DATA_PATH}/*)
# echo ${SEGMENTS}
let cnt=0
for SEGMENT in ${SEGMENTS}
do
    # echo $GPU
    for CAMERA in ${CAMERA_LIST}
    do
        if [ $GPU = $((${cnt} % 4)) ]; then
            echo $cnt
            echo ${SEGMENT}
            mkdir ${SEGMENT}/${CAMERA}/profile
            OUTPUT_PATH=${SEGMENT}/${CAMERA}/profile
            python3 create_youtube_video_input.py \
                --data_path=${SEGMENT}/${CAMERA} \
                --output_path=${OUTPUT_PATH}
            echo "Done creating input!"
            python3 infer_detections_for_ground_truth.py \
                --inference_graph=$MODEL_PATH \
                --discard_image_pixels \
                --gpu=$GPU \
                --input_tfrecord_paths=${OUTPUT_PATH}/input.record \
                --output_tfrecord_path=${OUTPUT_PATH}/gt_FasterRCNN_COCO.record \
                --output_time_path=${OUTPUT_PATH}/full_model_time_FasterRCNN_COCO.csv \
                --gt_csv=${OUTPUT_PATH}/gt_FasterRCNN_COCO.csv
            echo "Done inference on ${cnt} segnment!"
            rm ${OUTPUT_PATH}/input.record
        fi
    done
    let cnt=cnt+1
done
# for DATASET in $DATASET_LIST
# do
#     for SEGMENT in ${SEGMENT_ARRAY[${DATASET}]}
#     do
#         for CAMERA in ${CAMERA_LIST}
#         do
#             for RESIZE_RESOL in $RESIZE_RESOL_LIST
#             do
#                 # echo $RESIZE_RESOL
#                 # python3 resize.py \
#                 #     --dataset=${DATASET} \
#                 #     --resize_resol=$RESIZE_RESOL \
#                 #     --path=$DATA_PATH
#
#                 for QP in $QP_LIST
#                 do
#                     # python3 change_quantization.py \
#                     #     --dataset=$DATASET \
#                     #     --path=$DATA_PATH \
#                     #     --quality_parameter=$QP \
#                     #     --resolution=$RESIZE_RESOL
#
#                     # echo ${DATA_PATH}${DATASET}/${SEGMENT}/${CAMERA}
#                     # python3 ${CODE_PATH}/dataset_tools/create_youtube_video_input.py \
#                     #     --data_path=${DATA_PATH}${DATASET}/${SEGMENT}/${CAMERA}/${RESIZE_RESOL}/ \
#                     #     --output_path=${DATA_PATH}${DATASET}/${SEGMENT}/${CAMERA}/${RESIZE_RESOL}/profile/ \
#                     #     --metadata_file=$DATA_PATH${DATASET}/${SEGMENT}/${CAMERA}/metadata.json \
#                     #     --resize_resol=$RESIZE_RESOL
#
#                     # echo "Done creating input!"
#                     # python3 ${CODE_PATH}/inference/infer_detections_for_ground_truth.py \
#                     #     --inference_graph=$FULL_MODEL_PATH \
#                     #     --discard_image_pixels \
#                     #     --gpu=$GPU \
#                     #     --path=${DATA_PATH}${DATASET}/${SEGMENT}/${CAMERA}/${RESIZE_RESOL}/profile/ \
#                     #     --resize_resol=$RESIZE_RESOL
#
#                     # python3 infer_object_id_youtube.py \
#                     #     --path=${DATA_PATH}${DATASET}/${SEGMENT}/${CAMERA}/${RESIZE_RESOL}/profile/ \
#                     #     --metadata_file=$DATA_PATH${DATASET}/${SEGMENT}/${CAMERA}/metadata.json \
#                     #     --resize_resol=$RESIZE_RESOL
#
#                     python3 video_feature_youtube.py \
#                         --path=${DATA_PATH}${DATASET}/${SEGMENT}/${CAMERA}/${RESIZE_RESOL}/profile/ \
#                         --metadata_file=$DATA_PATH${DATASET}/${SEGMENT}/${CAMERA}/metadata.json \
#                         --dataset=${DATASET}_${SEGMENT}_${CAMERA}
#                 done
#             done
#         done
#     done
# done
