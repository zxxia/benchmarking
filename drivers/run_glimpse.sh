#!/bin/bash

VIDEOS='crossroad crossroad2 crossroad3 crossroad4 drift driving1 driving2
driving_downtown highway highway_normal_traffic nyc jp jp_hw lane_split
motorway park russia russia1 traffic tw tw1 tw_road tw_under_bridge road_trip'

# VIDEOS='chicago_virtual_run'
        # --frame_difference_threshold_divisor_list 20 15 10 8 5 2 \
# VIDEOS='tv_show'
#         --frame_difference_threshold_divisor_list 200 150 100 \
VIDEOS='london'

SHORT_VIDEO_LENGTH=30
# PROFILE_LENGTH=10
PROFILE_LENGTH=30


for VIDEO in $VIDEOS
do
    python glimpse \
        --video $VIDEO \
        --dataset youtube \
        --data_root /data/zxxia/videos \
        --short_video_length $SHORT_VIDEO_LENGTH \
        --profile_length $PROFILE_LENGTH \
        --overfitting \
        --frame_difference_threshold_divisor_list 20 15 10 8 5 2 \
        --output_filename glimpse_overfitting_results_${VIDEO}.csv \
        --profile_filename glimpse_overfitting_profile_${VIDEO}.csv
done
