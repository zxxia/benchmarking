#!/bin/bash
declare -A RESOL
declare -A NUM_OF_FRAMES


# Change CODE_PATH to the object detection path.
CODE_PATH="/home/zxxia/models/research/object_detection"
# Change full model path to path where trained object detection model is saved.
FULL_MODEL_PATH="/home/zxxia/models/research/"\
"object_detection/faster_rcnn_resnet101_coco_2018_01_28/"\
"frozen_inference_graph.pb"

# Change DATA_PATH to downloaded Youtube video path
DATA_PATH='/mnt/data/zhujun/new_video/' #'/home/zxxia/videos/'
# Change dataset_list to your dataset name
DATASET_LIST="drift tw1"
#"jp russia1 tw tw1 park"
# Choose an idle GPU
GPU="2"
RESIZE_RESOL=360p
RESIZE_FLAG=False
for DATASET_NAME in $DATASET_LIST
do
echo ${DATASET_NAME} 

# python3 resize.py \
#     --dataset=${DATASET_NAME} \
#     --resize_resol=$RESIZE_RESOL \
#     --path=$DATA_PATH
 
 #--frame_count=${NUM_OF_FRAMES[${DATASET_NAME}]} \
python3 ${CODE_PATH}/dataset_tools/create_youtube_video_input.py \
    --dataset=${DATASET_NAME} \
    --resize=$RESIZE_FLAG \
    --path=$DATA_PATH \
    --resize_resol=$RESIZE_RESOL

echo "Done creating input!"
python3  ${CODE_PATH}/inference/infer_detections_for_ground_truth.py \
	--inference_graph=$FULL_MODEL_PATH \
	--discard_image_pixels \
	--dataset=${DATASET_NAME} \
	--gpu=$GPU \
	--resize=$RESIZE_FLAG \
	--path=$DATA_PATH \
    --resize_resol=$RESIZE_RESOL

python3 infer_object_id_youtube.py \
	--dataset=${DATASET_NAME} \
 	--path=$DATA_PATH \
 	--resize=$RESIZE_FLAG \
    --resize_resol=$RESIZE_RESOL

python3 video_feature_youtube.py \
	--dataset=${DATASET_NAME} \
	--path=$DATA_PATH 

done

# Add resolution of new dataset
# RESOL=([walking]="3840,2160" \
# 	   [driving_downtown]="3840,2160" \
# 	   [highway]="1280,720" \
# 	   [crossroad2]="1920,1080" \
# 	   [crossroad]="1920,1080" \
# 	   [crossroad3]="1280,720" \
# 	   [crossroad4]="1920,1080" \
# 	   [crossroad5]="1920,1080" \
#        [driving1]="1920,1080" \
# 	   [driving2]="1280,720" \
# 	   [crossroad6]="1920,1080" \
# 	   [crossroad7]="1920,1080" \
# 	   [cropped_crossroad3]="600,400" \
# 	   [cropped_driving2]="600,400" \
# 	   [traffic]="1280,720" \
# 	   [highway_no_traffic]="1280,720" \
# 	   [highway_normal_traffic]="1280,720" \
#        [street_racing]="1280,720" \
#        [motor]="1280,720" \
#        [reckless_driving]="1280,720" \
#        [jp_hw]="1280,720" \
#        [jp]="1280,720" \
#        [russia]="1920,1080" \
#        [russia1]="1920,1080" \
#        [tw_road]="1280,720" \
#        [tw]="1280,720" \
#        [tw_under_bridge]="1280,720" \
#        [tw]="1280,720" \
#        [park]="1280,720")
# 
# # Add number of frames of new dataset
# NUM_OF_FRAMES=([walking]=68300\
# 	   [driving_downtown]=32000 \
# 	   [highway]=51200 \
# 	   [crossroad2]=85300 \
# 	   [crossroad]=58400 \
# 	   [crossroad3]=35800 \
# 	   [crossroad4]=36000 \
# 	   [crossroad5]=36000 \
#        [driving1]=32000 \
# 	   [driving2]=39000 \
# 	   [crossroad6]=36000 \
# 	   [crossroad7]=36000 \
# 	   [cropped_crossroad3]=36000 \
# 	   [cropped_driving2]=39000 \
#        [highway_normal_traffic]=1608 \
# 	   [traffic]=2348 \
# 	   [highway_no_traffic]=12378 \
#        [street_racing]=1296 \
#        [motor]=6394 \
#        [reckless_driving]=8219 \
#        [jp_hw]=9150 \
#        [jp]=36000 \
#        [russia]=9000 \
#        [russia1]=36000 \
#        [tw_road]=9150 \
#        [tw]=36150 \
#        [tw1]=36150 \
#        [tw_under_bridge]=9149 \
#        [park]=35847)


