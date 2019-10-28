#!/bin/bash
PROFILE_LENGTH=30
OFFSET=0
SHORT_VIDEO_LENGTH=30
TARGET_F1=0.9

# YouTube Videos
# OUTPUT_FILE="glimpse_E2E_youtube_result_to_delete.csv"
# DATA_PATH=/mnt/data/zhujun/dataset/Youtube/
DATA_PATH=/data/zxxia/videos/
VIDEOS="crossroad crossroad2 crossroad3 crossroad4 drift driving1 driving2
        driving_downtown traffic highway highway_normal_traffic jp
         motorway nyc park  russia1 traffic tw tw1
        tw_under_bridge" #lane_split russia
# VIDEOS=''
for VIDEO in $VIDEOS
do
    # python glimpse_E2E.py \
    #     --path $DATA_PATH$VIDEO/ \
    #     --video $VIDEO \
    #     --metadata $DATA_PATH$VIDEO/metadata.json \
    #     --output $OUTPUT_FILE \
    #     --short_video_length $SHORT_VIDEO_LENGTH \
    #     --profile_length $PROFILE_LENGTH \
    #     --offset $OFFSET \
    #     --target_f1 $TARGET_F1
    python glimpse_perfect_tracking.py \
        --path $DATA_PATH$VIDEO/ \
        --video $VIDEO \
        --metadata $DATA_PATH$VIDEO/metadata.json \
        --output glimpse_frame_select_results/glimpse_perfect_tracking_${VIDEO}.csv \
        --log glimpse_frame_select_profile/profile_${VIDEO}.csv \
        --short_video_length $SHORT_VIDEO_LENGTH \
        --profile_length $PROFILE_LENGTH \
        --offset $OFFSET \
        --target_f1 $TARGET_F1 &
done

# Waymo Videos
# DATA_PATH='/mnt/data/zhujun/new_video/'
# VIDEOS='training_0000 training_0001 training_0002 validation_0000'
# CAMERAS='FRONT FRONT_LEFT FRONT_RIGHT SIDE_LEFT SIDE_RIGHT'
# OUTPUT_FILE="glimpse_perfect_tracking_waymo_truth_label.csv"

# for VIDEO in $VIDEOS
# do
#     SEGMENT_FOLDERS=$(ls -d $DATA_PATH$VIDEO/*/)
#     # echo $SEGMENT_FOLDERS
#     for SEGMENT_FOLDER in $SEGMENT_FOLDERS
#     do
#         SEGMENT=$(echo ${SEGMENT_FOLDER%/}| sed -n -e 's/^.*\(segment\)/\1/p')
#         # echo $SEGMENT
#         # echo $SEGMENT_FOLDER
#         for CAMERA in $CAMERAS
#         do
#             VIDEO_NAME=$VIDEO\_$SEGMENT\_$CAMERA
#             echo $VIDEO_NAME
#             python glimpse_perfect_tracking_waymo.py \
#                 --path $SEGMENT_FOLDER$CAMERA/ \
#                 --video $VIDEO_NAME \
#                 --metadata $SEGMENT_FOLDER$CAMERA/metadata.json \
#                 --output $OUTPUT_FILE \
#                 --short_video_length $SHORT_VIDEO_LENGTH \
#                 --profile_length $PROFILE_LENGTH \
#                 --offset $OFFSET \
#                 --target_f1 $TARGET_F1
#             # python glimpse_perfect_tracking.py \
#             #     --path $SEGMENT_FOLDER$CAMERA/ \
#             #     --video $VIDEO_NAME \
#             #     --metadata $SEGMENT_FOLDER$CAMERA/metadata.json \
#             #     --output $OUTPUT_FILE \
#             #     --short_video_length $SHORT_VIDEO_LENGTH \
#             #     --profile_length $PROFILE_LENGTH \
#             #     --offset $OFFSET \
#             #     --target_f1 $TARGET_F1
#         done
#     done
# done


# kitti Videos
# KITII Videos do not have object ID
# DATA_PATH='/mnt/data/zhujun/dataset/KITTI/'
# VIDEOS='City Residential Road '
# OUTPUT_FILE="glimpse_perfect_tracking_kitti.csv"

# for VIDEO in $VIDEOS
# do
#     SEGMENT_FOLDERS=$(ls -d $DATA_PATH$VIDEO/*/)
#     # echo $SEGMENT_FOLDERS
#     for SEGMENT_FOLDER in $SEGMENT_FOLDERS
#     do
#         SEGMENT=$(echo ${SEGMENT_FOLDER%/}| sed -n -e 's/^.*\(2011\)/\1/p')
#         # echo $SEGMENT
#         # echo $SEGMENT_FOLDER
#         VIDEO_NAME=$VIDEO\_$SEGMENT
#         echo $VIDEO_NAME
#         python glimpse_perfect_tracking.py \
#             --path ${SEGMENT_FOLDER}image_02\/data\/ \
#             --video $VIDEO_NAME \
#             --output $OUTPUT_FILE \
#             --short_video_length $SHORT_VIDEO_LENGTH \
#             --profile_length $PROFILE_LENGTH \
#             --offset $OFFSET \
#             --target_f1 $TARGET_F1 \
#             --fps 10 \
#             --resolution 1242 375 \
#             --format {:010d}.png
#         break
#     done
#     break
# done
