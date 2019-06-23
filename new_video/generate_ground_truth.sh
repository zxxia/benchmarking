#!/bin/bash
declare -A RESOL
declare -A NUM_OF_FRAMES
# Add resolution of new dataset
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

# Add number of frames of new dataset
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



# Change CODE_PATH to the object detection path.
CODE_PATH="/home/zhujun/video_analytics_pipelines/models/research/object_detection"
# Change full model path to path where trained object detection model is saved.
FULL_MODEL_PATH="/home/zhujun/video_analytics_pipelines/models/research/"\
"object_detection/faster_rcnn_resnet101_coco_2018_01_28/"\
"frozen_inference_graph.pb"

# Change DATA_PATH to downloaded Youtube video path
DATA_PATH='/home/zhujun/video_analytics_pipelines/dataset/Youtube/'
# Change dataset_list to your dataset name
DATASET_LIST="crossroad2"
# Choose an idle GPU
GPU="1"



for DATASET_NAME in $DATASET_LIST
do
echo ${DATASET_NAME} 


python3 ${CODE_PATH}/dataset_tools/create_youtube_video_input.py \
		--dataset=${DATASET_NAME} \
		--resol=${RESOL[${DATASET_NAME}]} \
		--frame_count=${NUM_OF_FRAMES[${DATASET_NAME}]} \
		--resize=False \
		--path=$DATA_PATH \

echo "Done creating input!"
python3  ${CODE_PATH}/inference/infer_detections_for_ground_truth.py \
		--inference_graph=$FULL_MODEL_PATH \
		--discard_image_pixels \
		--dataset=${DATASET_NAME} \
		--gpu=$GPU \
		--resize=False \
		--path=$DATA_PATH \

python3 infer_object_id_youtube.py \
		--dataset=${DATASET_NAME} \
		--path=$DATA_PATH \
		--resize=False \
		--resol=${RESOL[${DATASET_NAME}]} \

python3 video_feature_youtube.py \
		--dataset=${DATASET_NAME} \
		--path=$DATA_PATH \
		--resol=${RESOL[${DATASET_NAME}]} 

done



