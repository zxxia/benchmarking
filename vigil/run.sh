#!/bin/bash

PROFILE_LENGTH=30
OFFSET=0
SHORT_VIDEO_LENGTH=30
TARGET_F1=0.9

# YouTube Videos
# DATA_PATH=/mnt/data/zhujun/dataset/Youtube/
DATA_PATH=/data/zxxia/videos/

# VIDEOS='crossroad crossroad2 crossroad3 crossroad4 drift driving1 driving2
# driving_downtown highway highway_normal_traffic jp lane_split motorway nyc
# park russia russia1 tw tw_road tw_under_bridge traffic'
VIDEOS='motorway'
for VIDEO in $VIDEOS
do
    python vigil_overfitting.py \
        --video $VIDEO \
        --metadata ${DATA_PATH}${VIDEO}/metadata.json \
        --output ~/Projects/benchmarking/vigil/vigil_overfitting_results_plus_08/vigil_${VIDEO}.csv\
        --profile_length ${PROFILE_LENGTH} \
        --short_video_length ${SHORT_VIDEO_LENGTH} \
        --save_path /data/zxxia/benchmarking/Vigil/masked_images_plus_08 \
        --resize_percent 0.8
        # --mask_frames \
done
# VIDEOS='crossroad crossroad2 crossroad3 crossroad4 drift driving1 driving2
# driving_downtown highway highway_normal_traffic jp lane_split motorway nyc
# park russia russia1 tw tw_road tw_under_bridge traffic'
# for VIDEO in $VIDEOS
# do
#     python vigil_e2e.py \
#         --video $VIDEO \
#         --metadata ${DATA_PATH}${VIDEO}/metadata.json \
#         --output ~/Projects/benchmarking/vigil/vigil_e2e_results/vigil_${VIDEO}.csv\
#         --profile_length ${PROFILE_LENGTH} \
#         --short_video_length ${SHORT_VIDEO_LENGTH} \
#         --save_path /data/zxxia/benchmarking/Vigil/masked_images #\
#         # --mask_frames
# done
