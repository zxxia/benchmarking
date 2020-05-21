#!/bin/bash

VIDEOS='crossroad crossroad2 crossroad3 crossroad4 drift driving1 driving2
driving_downtown highway highway_normal_traffic nyc jp jp_hw lane_split
motorway park russia russia1 traffic tw tw1 tw_road tw_under_bridge road_trip'

VIDEOS='chicago_virtual_run'
# VIDEOS='tv_show'
# VIDEOS='london'

SHORT_VIDEO_LENGTH=30
# PROFILE_LENGTH=10
PROFILE_LENGTH=30


for VIDEO in $VIDEOS
do
    python vigil \
        --video $VIDEO \
        --dataset youtube \
        --data_root /data/zxxia/videos \
        --short_video_length $SHORT_VIDEO_LENGTH \
        --profile_length $PROFILE_LENGTH \
        --crop \
        --output_filename vigil_spatial_overfitting_results_${VIDEO}.csv \
        --video_save_path .
        # --profile_filename vigil_spatial_overfitting_profile_${VIDEO}.csv \
done