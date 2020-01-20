#!/bin/bash

PROFILE_LENGTH=10
OFFSET=0
SHORT_VIDEO_LENGTH=30
TARGET_F1=0.9

# YouTube Videos
DATA_PATH='/mnt/data/zhujun/dataset/Youtube/'
# DATA_PATH=/data/zxxia/videos/

# VIDEOS='crossroad crossroad2 crossroad3 crossroad4 drift driving1 driving2
# driving_downtown highway highway_normal_traffic jp lane_split motorway nyc
# park russia russia1 tw tw_road tw_under_bridge traffic'
VIDEOS='crossroad2_night'
for VIDEO in $VIDEOS
do
    # python vigil_overfitting.py \
    #     --path ${DATA_PATH} \
    #     --video ${VIDEO} \
    #     --metadata ${DATA_PATH}${VIDEO}/metadata.json \
    #     --output vigil_mobilenet_results_10_11/vigil_${VIDEO}.csv\
    #     --short_video_length ${SHORT_VIDEO_LENGTH} \
    #     --offset ${OFFSET} &
    # python vigil_baseline.py \
    #     --path ${DATA_PATH} \
    #     --video ${VIDEO} \
    #     --metadata ${DATA_PATH}${VIDEO}/metadata.json \
    #     --output baseline_results/vigil_${VIDEO}.csv \
    #     --short_video_length ${SHORT_VIDEO_LENGTH} \
    #     --offset ${OFFSET}
    python vigil_e2e.py \
        --video $VIDEO \
        --metadata ${DATA_PATH}${VIDEO}/metadata.json \
        --output /mnt/data/zhujun/dataset/Vigil_result/vigil_${VIDEO}.csv\
        --profile_length ${PROFILE_LENGTH} \
        --short_video_length ${SHORT_VIDEO_LENGTH} \
        --save_path /mnt/data/zhujun/dataset/Vigil_result/ \
        --mask_frames
done
        # --path ${DATA_PATH} \
