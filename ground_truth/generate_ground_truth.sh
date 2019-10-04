#!/bin/bash
declare -A RESOL
declare -A NUM_OF_FRAMES


# Change CODE_PATH to the object detection path.
CODE_PATH="/home/zxxia/models/research/object_detection"
# Change full model path to path where trained object detection model is saved.
FULL_MODEL_PATH="/home/zxxia/models/research/"\
"object_detection/faster_rcnn_resnet101_coco_2018_01_28/"\
"frozen_inference_graph.pb"

MOBILENET_SSD_PATH="/home/zxxia/models/research/"\
"object_detection/ssd_mobilenet_v2_coco_2018_03_29/"\
"frozen_inference_graph.pb"

# Change DATA_PATH to downloaded Youtube video path
DATA_PATH='/data/zxxia/benchmarking/results/videos/'
DATA_PATH_tmp='/data/zxxia/videos/'
# DATASET_LIST="crossroad crossroad2 crossroad3 crossroad4 drift"
# DATASET_LIST="driving1 driving2 driving_downtown highway_normal_traffic"
# DATASET_LIST="jp jp_hw motorway nyc park"
# DATASET_LIST="russia russia1 traffic tw tw1 tw_road tw_under_bridge"
DATASET_LIST="crossroad2 crossroad3 crossroad4 crossroad highway driving1 driving2 driving_downtown jp lane_split nyc motorway park russia1 russia t_crossroad traffic tw tw1 tw_road tw_under_bridge"

# Choose an idle GPU
GPU="1"
RESOL_LIST='720p' # 540p 480p 360p'
QP_LIST="original" #'30 35 40'
for DATASET in $DATASET_LIST
do
    echo ${DATASET_NAME}

    for RESOL in $RESOL_LIST
    do
        # echo $RESIZE_RESOL
        # python3 resize.py \
        #     --dataset=${DATASET_NAME} \
        #     --resize_resol=$RESIZE_RESOL \
        #     --path=$DATA_PATH

        for QP in $QP_LIST
        do
            # python3 change_quantization.py \
            #     --dataset=$DATASET_NAME \
            #     --path=$DATA_PATH \
            #     --quality_parameter=$QP \
            #     --resolution=$RESIZE_RESOL

            # python3 ${CODE_PATH}/dataset_tools/create_youtube_video_input.py \
            #     --metadata_file=${DATA_PATH}/${DATASET}/metadata.json \
            #     --data_path=${DATA_PATH}${DATASET}/${RESOL} \
            #     --output_path=${DATA_PATH}${DATASET}/${RESOL}/profile/ \
            #     --resol=$RESOL

            echo "Done creating input!"
            # Run inference on mobilenet ssd
            # python3 ${CODE_PATH}/inference/infer_detections_for_ground_truth.py \
            #     --inference_graph=$MOBILENET_SSD_PATH \
            #     --discard_image_pixels \
            #     --gpu=$GPU \
            #     --input_tfrecord_paths=${DATA_PATH}${DATASET}/${RESOL}/profile/input.record \
            #     --output_tfrecord_path=${DATA_PATH}${DATASET}/${RESOL}/profile/gt_mobilenet_COCO.record \
            #     --output_time_path=${DATA_PATH}${DATASET}/${RESOL}/profile/full_model_time_mobilenet_COCO.csv \
            #     --gt_csv=${DATA_PATH}${DATASET}/${RESOL}/profile/gt_mobilenet_COCO.csv

            # Run inferenece on full model FasterRCNN
 #           python3 ${CODE_PATH}/inference/infer_detections_for_ground_truth.py \
 #               --inference_graph=$FULL_MODEL_PATH \
 #               --discard_image_pixels \
 #               --gpu=$GPU \
 #               --input_tfrecord_paths=${DATA_PATH}${DATASET}/${RESOL}/profile/input.record \
 #               --output_tfrecord_path=${DATA_PATH}${DATASET}/${RESOL}/profile/gt_FasterRCNN_COCO.record \
 #               --output_time_path=${DATA_PATH}${DATASET}/${RESOL}/profile/full_model_time_FasterRCNN_COCO.csv \
 #               --gt_csv=${DATA_PATH}${DATASET}/${RESOL}/profile/gt_FasterRCNN_COCO.csv

#            python3 infer_object_id.py \
#                --resol=$RESOL \
#                --input_file=${DATA_PATH}${DATASET}/${RESOL}/profile/gt_FasterRCNN_COCO.csv \
#                --output_file=${DATA_PATH}/${DATASET}/720p/profile/updated_gt_FasterRCNN_COCO_no_filter.csv

            python3 video_feature_youtube.py \
                --metadata_file=${DATA_PATH_tmp}/${DATASET}/metadata.json \
                --input_file=${DATA_PATH}/${DATASET}/720p/profile/updated_gt_FasterRCNN_COCO_no_filter.csv \
                --output_file=${DATA_PATH}/${DATASET}/720p/profile/Video_features_${DATASET}_object_type_filter.csv \
                --resol=${RESOL}
        done
    done
done






#"08041754 08041847 08041951 08042133 08050938 08051137"
#"akiyo_cif deadline_cif ice_4cif pedestrian_area_1080p25 "\
#"bowing_cif football_422_ntsc KristenAndSara_1280x720_60 "\
#"rush_hour_1080p25 bus_cif foreman_cif mad900_cif "\
#"station2_1080p25 carphone_qcif FourPeople_1280x720_60 "\
#"miss_am_qcif tractor_1080p25 claire_qcif grandma_qcif "\
#"mthr_dotr_qcif coastguard_cif hall_objects_qcif "\
#"Netflix_Crosswalk_4096x2160_60fps_10bit_420 crew_4cif "\
#"highway_cif Netflix_DrivingPOV_4096x2160_60fps_10bit_420"
