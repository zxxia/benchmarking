#!/bin/bash
declare -A RESOL
declare -A NUM_OF_FRAMES


# Change CODE_PATH to the object detection path.
# CODE_PATH="/home/zxxia/models/research/object_detection"
CODE_PATH="/data/zxxia/models/research/object_detection"
# Change full model path to path where trained object detection model is saved.
# FULL_MODEL_PATH="/home/zxxia/models/research/"\
# "object_detection/faster_rcnn_resnet101_coco_2018_01_28/"\
# "frozen_inference_graph.pb"
FULL_MODEL_PATH="/data/zxxia/models/research/"\
"object_detection/faster_rcnn_resnet101_coco_2018_01_28/"\
"frozen_inference_graph.pb"

# MOBILENET_SSD_PATH="/home/zxxia/models/research/"\
# "object_detection/ssd_mobilenet_v2_coco_2018_03_29/"\
# "frozen_inference_graph.pb"
MOBILENET_SSD_PATH="/data/zxxia/models/research/"\
"object_detection/ssd_mobilenet_v2_coco_2018_03_29/"\
"frozen_inference_graph.pb"

# Change DATA_PATH to downloaded Youtube video path
# DATA_PATH='/data/zxxia/benchmarking/results/videos/'
DATA_PATH='/data/zxxia/videos/'
DATASET_LIST="crossroad crossroad2 crossroad3 crossroad4 driving1 driving2
              drift driving_downtown highway highway_normal_traffic jp
              lane_split motorway nyc park russia russia1 tw tw1 tw_under_bridge tw_road traffic"

RESULT_PATH='/data/zxxia/benchmarking/results/videos/'
# DATASET_LIST="road_trip"
declare -A VIDEO_TYPE_ARRAY
VIDEO_TYPE_ARRAY["driving1"]="moving"
VIDEO_TYPE_ARRAY["driving2"]="moving"
VIDEO_TYPE_ARRAY["driving_downtown"]="moving"
VIDEO_TYPE_ARRAY["park"]="moving"
VIDEO_TYPE_ARRAY["lane_split"]="static"
VIDEO_TYPE_ARRAY["crossroad"]="static"
VIDEO_TYPE_ARRAY["crossroad2"]="static"
VIDEO_TYPE_ARRAY["crossroad3"]="static"
VIDEO_TYPE_ARRAY["crossroad4"]="static"
VIDEO_TYPE_ARRAY["drift"]="static"
VIDEO_TYPE_ARRAY["highway"]="static"
VIDEO_TYPE_ARRAY["highway_normal_traffic"]="static"
VIDEO_TYPE_ARRAY["jp"]="static"
VIDEO_TYPE_ARRAY["jp_hw"]="static"
VIDEO_TYPE_ARRAY["motorway"]="static"
VIDEO_TYPE_ARRAY["nyc"]="static"
VIDEO_TYPE_ARRAY["russia"]="static"
VIDEO_TYPE_ARRAY["russia1"]="static"
VIDEO_TYPE_ARRAY["traffic"]="static"
VIDEO_TYPE_ARRAY["tw"]="static"
VIDEO_TYPE_ARRAY["tw1"]="static"
VIDEO_TYPE_ARRAY["tw_road"]="static"
VIDEO_TYPE_ARRAY["tw_under_bridge"]="static"
VIDEO_TYPE_ARRAY["t_crossroad"]="static"
VIDEO_TYPE_ARRAY["road_trip"]="moving"


# Choose an idle GPU
GPU="1"
RESOL_LIST='180p' #1080p 720p 540p 480p 360p 300p'
QP_LIST="original" #'30 35 40'
for DATASET in $DATASET_LIST
do
    echo ${DATASET}

    for RESOL in $RESOL_LIST
    do
        # echo $RESOL
        # echo ${DATA_PATH}${DATASET}/${RESOL}
        # mkdir ${DATA_PATH}${DATASET}/${RESOL}
        # python3 resize.py \
        #     --input_video=${DATA_PATH}${DATASET}/${DATASET}.mp4 \
        #     --output_video=${DATA_PATH}${DATASET}/${RESOL}/${DATASET}_${RESOL}.mp4\
        #     --output_image_path=${DATA_PATH}${DATASET}/${RESOL}/ \
        #     --resol=$RESOL

        for QP in $QP_LIST
        do
            # python3 change_quantization.py \
            #     --dataset=$DATASET_NAME \
            #     --path=$DATA_PATH \
            #     --quality_parameter=$QP \
            #     --resolution=$RESIZE_RESOL

            # python3 ${CODE_PATH}/dataset_tools/create_youtube_video_input.py \
            #     --metadata_file=${DATA_PATH}${DATASET}/metadata.json \
            #     --data_path=${DATA_PATH}${DATASET}/${RESOL} \
            #     --output_path=${DATA_PATH}${DATASET}/${RESOL}/profile/ \
            #     --resol=$RESOL
            #
            # echo "Done creating input!"

            # Run inferenece on full model FasterRCNN
            # python3 ${CODE_PATH}/inference/infer_detections_for_ground_truth.py \
            #     --inference_graph=$FULL_MODEL_PATH \
            #     --discard_image_pixels \
            #     --gpu=$GPU \
            #     --input_tfrecord_paths=${DATA_PATH}${DATASET}/${RESOL}/profile/input.record \
            #     --output_tfrecord_path=${DATA_PATH}${DATASET}/${RESOL}/profile/gt_FasterRCNN_COCO.record \
            #     --output_time_path=${DATA_PATH}${DATASET}/${RESOL}/profile/full_model_time_FasterRCNN_COCO.csv \
            #     --gt_csv=${DATA_PATH}${DATASET}/${RESOL}/profile/gt_FasterRCNN_COCO.csv
            # echo "Done inference!"

            # python3 infer_object_id.py \
            #     --resol=$RESOL \
            #     --input_file=${DATA_PATH}${DATASET}/${RESOL}/profile/gt_FasterRCNN_COCO.csv \
            #     --output_file=${DATA_PATH}/${DATASET}/${RESOL}/profile/updated_gt_FasterRCNN_COCO_no_filter.csv
            # mkdir /data/zxxia/benchmarking/results/videos/${DATASET}/${RESOL}
            # mkdir /data/zxxia/benchmarking/results/videos/${DATASET}/${RESOL}/profile
            # cp ${DATA_PATH}/${DATASET}/${RESOL}/profile/*.csv /data/zxxia/benchmarking/results/videos/${DATASET}/${RESOL}/profile/

            # python3 video_feature_youtube.py \
            #     --metadata_file=${DATA_PATH}/${DATASET}/metadata.json \
            #     --input_file=${DATA_PATH}/${DATASET}/${RESOL}/profile/updated_gt_FasterRCNN_COCO_no_filter.csv \
            #     --output_file=${RESULT_PATH}/${DATASET}/${RESOL}/profile/Video_features_${DATASET}_object_width_height_type_filter.csv \
            #     --video_type=${VIDEO_TYPE_ARRAY[${DATASET}]} \
            #     --resol=${RESOL}
        done
    done
done



# for DATASET in $DATASET_LIST
# do
#     echo ${DATASET_NAME}
#
#     for RESOL in $RESOL_LIST
#     do
#         # echo $RESIZE_RESOL
#         # python3 resize.py \
#         #     --dataset=${DATASET_NAME} \
#         #     --resize_resol=$RESIZE_RESOL \
#         #     --path=$DATA_PATH
#
#         for QP in $QP_LIST
#         do
#             # python3 change_quantization.py \
#             #     --dataset=$DATASET_NAME \
#             #     --path=$DATA_PATH \
#             #     --quality_parameter=$QP \
#             #     --resolution=$RESIZE_RESOL
#
#             # python3 ${CODE_PATH}/dataset_tools/create_youtube_video_input.py \
#             #     --metadata_file=${DATA_PATH}${DATASET}/metadata.json \
#             #     --data_path=${DATA_PATH}${DATASET}/${RESOL} \
#             #     --output_path=${DATA_PATH}${DATASET}/${RESOL}/profile/ \
#             #     --resol=$RESOL
#             #
#             # echo "Done creating input!"
#             # Run inference on mobilenet ssd
#             # python3 ${CODE_PATH}/inference/infer_detections_for_ground_truth.py \
#             #     --inference_graph=$MOBILENET_SSD_PATH \
#             #     --discard_image_pixels \
#             #     --gpu=$GPU \
#             #     --input_tfrecord_paths=${DATA_PATH}${DATASET}/${RESOL}/profile/input.record \
#             #     --output_tfrecord_path=${DATA_PATH}${DATASET}/${RESOL}/profile/gt_mobilenet_COCO.record \
#             #     --output_time_path=${DATA_PATH}${DATASET}/${RESOL}/profile/full_model_time_mobilenet_COCO.csv \
#             #     --gt_csv=${DATA_PATH}${DATASET}/${RESOL}/profile/gt_mobilenet_COCO.csv
#             #
#             # echo "Done inference!"
#             python3 infer_object_id.py \
#                 --resol=$RESOL \
#                 --input_file=${DATA_PATH}${DATASET}/${RESOL}/profile/gt_mobilenet_COCO.csv \
#                 --output_file=${RESULT_PATH}/${DATASET}/${RESOL}/profile/updated_gt_mobilenet_COCO_no_filter.csv \
#                 --model_name=MobilenetSSD
#
#             python3 video_feature_youtube.py \
#                 --metadata_file=${DATA_PATH}/${DATASET}/metadata.json \
#                 --input_file=${RESULT_PATH}/${DATASET}/${RESOL}/profile/updated_gt_mobilenet_COCO_no_filter.csv \
#                 --output_file=${RESULT_PATH}/${DATASET}/${RESOL}/profile/Video_features_${DATASET}_object_width_height_type_filter_mobilenet.csv \
#                 --video_type=${VIDEO_TYPE_ARRAY[${DATASET}]} \
#                 --resol=${RESOL}
#         done
#     done
# done
