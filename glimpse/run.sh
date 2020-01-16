#!/bin/bash
PROFILE_LENGTH=10
OFFSET=0
SHORT_VIDEO_LENGTH=30
TARGET_F1=0.9

# YouTube Videos
DATA_PATH=/data/zxxia/videos
VIDEOS="crossroad crossroad2 crossroad3 crossroad4 drift driving1 driving2
        driving_downtown traffic highway highway_normal_traffic jp
         motorway nyc park russia1 traffic tw tw1
        tw_under_bridge lane_split russia"
# VIDEOS="highway"
VIDEOS="motorway highway park"

# for VIDEO in $VIDEOS
# do
#     python test_glimpse_frame_select.py \
#         --video $VIDEO \
#         --metadata $DATA_PATH$VIDEO/metadata.json \
#         --output glimpse_e2e_frame_select_results/glimpse_frame_select_${VIDEO}.csv \
#         --log glimpse_e2e_frame_select_profile/profile_${VIDEO}.csv \
#         --short_video_length $SHORT_VIDEO_LENGTH \
#         --profile_length $PROFILE_LENGTH \
#         --offset $OFFSET \
#         --trace_path frame_select_traces &
#         # --target_f1 $TARGET_F1 &
# done

for VIDEO in $VIDEOS
do
    # /home/zxxia/cpulimit-master/src/cpulimit -l 100 python glimpse_e2e.py \
    python glimpse_e2e.py \
        --video $VIDEO \
        --metadata $DATA_PATH/$VIDEO/metadata.json \
        --output /data/zxxia/benchmarking/glimpse/glimpse_e2e_results/glimpse_result_${VIDEO}.csv \
        --log /data/zxxia/benchmarking/glimpse/glimpse_e2e_results/glimpse_profile_${VIDEO}.csv \
        --short_video_length $SHORT_VIDEO_LENGTH \
        --profile_length $PROFILE_LENGTH \
        --offset $OFFSET \
        --trace_path /data/zxxia/benchmarking/glimpse/glimpse_e2e_results \
        --profile_trace_path /data/zxxia/benchmarking/glimpse/glimpse_e2e_results
        # --target_f1 $TARGET_F1 &
done

# DATA_PATH='/data2/zxxia/KITTI/'
# RESULT_PATH='/data/zxxia/benchmarking/results/KITTI/'
# LOCATIONS="City Residential Road"
# for LOCATION in ${LOCATIONS}
# do
#     SEGMENTS=$(cd ${DATA_PATH}/${LOCATION} && ls -d */)
#
#     for SEGMENT in ${SEGMENTS}
#     do
#         let end=$(ls -l ${DATA_PATH}/${LOCATION}/${SEGMENT}/image_02/data/375p/*.png | wc -l)
#         let start=0
#         let end--
#         let duration="($end/10)+1"
#         echo $start $end $duration
#         VIDEO=kitti_${LOCATION}_${SEGMENT//\/}
#         echo ${video}
#         python glimpse_frame_select.py \
#             --path $DATA_PATH$VIDEO/ \
#             --video $VIDEO \
#             --metadata $DATA_PATH$VIDEO/metadata.json \
#             --output glimpse_frame_select_results_kitti/glimpse_perfect_tracking_${VIDEO}.csv \
#             --log glimpse_frame_select_profile_kitti/profile_${VIDEO}.csv \
#             --short_video_length $SHORT_VIDEO_LENGTH \
#             --profile_length $PROFILE_LENGTH \
#             --offset $OFFSET \
#             --target_f1 $TARGET_F1 &
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


