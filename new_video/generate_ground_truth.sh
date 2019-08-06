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
DATA_PATH='/mnt/data/zhujun/dataset/Youtube/canada_crossroad/' 
# Change dataset_list to your dataset name
DATASET_LIST="08041754 08041847 08041951 08042133 08050938 08051137"
#"akiyo_cif deadline_cif ice_4cif pedestrian_area_1080p25 "\
#"bowing_cif football_422_ntsc KristenAndSara_1280x720_60 "\
#"rush_hour_1080p25 bus_cif foreman_cif mad900_cif "\
#"station2_1080p25 carphone_qcif FourPeople_1280x720_60 "\
#"miss_am_qcif tractor_1080p25 claire_qcif grandma_qcif "\
#"mthr_dotr_qcif coastguard_cif hall_objects_qcif "\
#"Netflix_Crosswalk_4096x2160_60fps_10bit_420 crew_4cif "\
#"highway_cif Netflix_DrivingPOV_4096x2160_60fps_10bit_420"  
#"jp highway motorway" # driving_downtown crossroad4 crossroad2 crossroad driving1 russia russia1" 
#"jp russia1 tw tw1 park"
# Choose an idle GPU
GPU="0"
RESIZE_RESOL_LIST='original' # 540p 480p 360p'
RESIZE_FLAG=Flase
QP_LIST="original" #'30 35 40'
for DATASET_NAME in $DATASET_LIST
do
    echo ${DATASET_NAME} 

    for RESIZE_RESOL in $RESIZE_RESOL_LIST
    do 
        echo $RESIZE_RESOL
        #if [ "$RESIZE_FLAG" = True ]; then
        # python3 resize.py \
        #     --dataset=${DATASET_NAME} \
        #     --resize_resol=$RESIZE_RESOL \
        #     --path=$DATA_PATH
        #fi 
            
        for QP in $QP_LIST
        do 
            python3 change_quantization.py \
                --dataset=$DATASET_NAME \
                --path=$DATA_PATH \
                --quality_parameter=$QP \
                --resolution=$RESIZE_RESOL

            python3 ${CODE_PATH}/dataset_tools/create_youtube_video_input.py \
                --dataset=${DATASET_NAME} \
                --path=$DATA_PATH \
                --resize_resol=$RESIZE_RESOL \
                --quality_parameter=$QP
            
            echo "Done creating input!"
             python3  ${CODE_PATH}/inference/infer_detections_for_ground_truth.py \
                 --inference_graph=$FULL_MODEL_PATH \
                 --discard_image_pixels \
                 --dataset=${DATASET_NAME} \
                 --gpu=$GPU \
                 --path=$DATA_PATH \
                 --resize_resol=$RESIZE_RESOL \
                 --quality_parameter=$QP
            
            python3 infer_object_id_youtube.py \
                --dataset=${DATASET_NAME} \
                --path=$DATA_PATH \
                --resize_resol=$RESIZE_RESOL \
                --quality_parameter=$QP
            
            if [ "$RESIZE_RESOL" = Original ]; then
                python3 video_feature_youtube.py \
                    --dataset=${DATASET_NAME} \
                    --path=$DATA_PATH 
            fi
        done
    done

done
