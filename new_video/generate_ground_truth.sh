#!/bin/bash
declare -A RESOL
declare -A NUM_OF_FRAMES
RESOL=([walking]="3840,2160" \
	   [driving_downtown]="3840,2160" \
	   [highway]="1280,720" \
	   [crossroad2]="1920,1080" \
	   [crossroad]="1920,1080" \
	   [crossroad3]="1280,720" \
	   [crossroad4]="1920,1080" \
	   [crossroad5]="1920,1080" \
       [driving1]="1920,1080" \
	   [driving2]="1280,720" \
	   [crossroad6]="1920,1080" \
	   [crossroad7]="1920,1080" \
	   [cropped_crossroad3]="600,400" \
	   [cropped_driving2]="600,400")

NUM_OF_FRAMES=([walking]=68300\
	   [driving_downtown]=32000 \
	   [highway]=51200 \
	   [crossroad2]=85300 \
	   [crossroad]=58400 \
	   [crossroad3]=35800 \
	   [crossroad4]=36000 \
	   [crossroad5]=36000 \
       [driving1]=32000 \
	   [driving2]=39000 \
	   [crossroad6]=36000 \
	   [crossroad7]=36000 \
	   [cropped_crossroad3]=36000 \
	   [cropped_driving2]=39000)


CODE_PATH="/home/zhujun/video_analytics_pipelines/models/research/object_detection"




# # FULL_MODEL_PATH="/home/zhujun/video_analytics_pipelines/models/research/"\
# # "object_detection/faster_rcnn_resnet101_kitti_2018_01_28/"\
# # "frozen_inference_graph.pb"


FULL_MODEL_PATH="/home/zhujun/video_analytics_pipelines/models/research/"\
"object_detection/faster_rcnn_resnet101_coco_2018_01_28/"\
"frozen_inference_graph.pb"

# #"object_detection/faster_rcnn_resnet101_coco_2018_01_28/"

# "object_detection/ssd_mobilenet_v2_coco_2018_03_29/"\

# DATASET_LIST="highway driving_downtown crossroad crossroad2 crossroad3 crossroad4 crossroad5 crossroad6 crossroad7 walking driving1 driving2"\
# " cropped_crossroad3 cropped_driving2"
DATA_PATH='/home/zhujun/video_analytics_pipelines/dataset/Youtube/'
DATASET_LIST="crossroad2 crossroad4 crossroad5"
GPU="1"
RESIZE_RESOL=540p



for DATASET_NAME in $DATASET_LIST
do
echo ${DATASET_NAME} 

python3 resize.py \
		--dataset=${DATASET_NAME} \
		--frame_count=${NUM_OF_FRAMES[${DATASET_NAME}]} \
		--resize_resol=$RESIZE_RESOL \
		--path=$DATA_PATH

echo 'Done resizing.'
python3 ${CODE_PATH}/dataset_tools/create_youtube_video_input.py \
		--dataset=${DATASET_NAME} \
		--resol=${RESOL[${DATASET_NAME}]} \
		--frame_count=${NUM_OF_FRAMES[${DATASET_NAME}]} \
		--resize=True \
		--path=$DATA_PATH \
		--resize_resol=$RESIZE_RESOL

echo "Done creating input!"
python3  ${CODE_PATH}/inference/infer_detections_for_ground_truth.py \
		--inference_graph=$FULL_MODEL_PATH \
		--discard_image_pixels \
		--dataset=${DATASET_NAME} \
		--gpu=$GPU \
		--resize=True \
		--path=$DATA_PATH \
		--resize_resol=$RESIZE_RESOL
# # do
python3 infer_object_id_youtube.py \
		--dataset=${DATASET_NAME} \
		--path=$DATA_PATH \
		--resize=True \
		--resol=${RESOL[${DATASET_NAME}]} \
		--resize_resol=$RESIZE_RESOL

# python3 ../dataset_profile/video_feature_youtube.py \
# 		--dataset=${DATASET_NAME} \
# 		--path=$DATA_PATH \
# 		--resol=${RESOL[${DATASET_NAME}]} 

done



