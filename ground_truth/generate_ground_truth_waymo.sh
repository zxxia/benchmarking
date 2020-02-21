#!/bin/bash

# Change CODE_PATH to the object detection path.
CODE_PATH="/home/zxxia/models/research/object_detection"
# Change full model path to path where trained object detection model is saved.

MODEL_PATH=/data/zxxia/ekya/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb
# Change DATA_PATH to downloaded Youtube video path
# DATA_PATH='/mnt/data/zhujun/dataset/Youtube/'
DATA_PATH='/data/zxxia/ekya/datasets/waymo_images_detection_results'
# DATA_PATH='/data/zxxia/ekya/datasets/waymo_images'
# Change dataset_list to your dataset name

# CAMERA_LIST="FRONT FRONT_LEFT FRONT_RIGHT SIDE_LEFT SIDE_RIGHT"
CAMERA_LIST="FRONT"

GPU=0
RESOL='360p' # 480p 360p' #'720p' #
QP_LIST="original" #'30 35 40'

SEGMENTS=$(ls -d ${DATA_PATH}/*)
echo ${SEGMENTS}
let cnt=0
for SEGMENT in ${SEGMENTS}
do
    # echo $GPU
    for CAMERA in ${CAMERA_LIST}
    do
        # if [ $GPU = $((${cnt} % 4)) ]; then
            # echo $cnt
            echo ${SEGMENT}
            # mkdir ${SEGMENT}/${CAMERA}/profile
            # OUTPUT_PATH=${SEGMENT}/${CAMERA}/profile
            # python3 create_youtube_video_input.py \
            #     --data_path=${SEGMENT}/${CAMERA} \
            #     --output_path=/home/zxxia/Projects/ekya/src/inference \
            #     --dataset=waymo
                # --output_path=${OUTPUT_PATH}
            # echo "Done creating input!"
            # python3 infer_detections_for_ground_truth.py \
            #     --inference_graph=$MODEL_PATH \
            #     --discard_image_pixels \
            #     --gpu=$GPU \
            #     --input_tfrecord_paths=${OUTPUT_PATH}/input.record \
            #     --output_tfrecord_path=${OUTPUT_PATH}/gt_FasterRCNN_COCO.record \
            #     --output_time_path=${OUTPUT_PATH}/full_model_time_FasterRCNN_COCO.csv \
            #     --gt_csv=${OUTPUT_PATH}/gt_FasterRCNN_COCO.csv
            # echo "Done inference on ${cnt} segnment!"
            # rm ${OUTPUT_PATH}/input.record
            python3 infer_object_id.py \
                --resol=$RESOL \
                --input_file=${SEGMENT}/${CAMERA}/profile/gt_FasterRCNN_COCO_$RESOL.csv \
                --output_file=${SEGMENT}/${CAMERA}/profile/updated_gt_FasterRCNN_COCO_no_filter_$RESOL.csv &
            # break
        # fi
    done
    let cnt=cnt+1
    # break
done
