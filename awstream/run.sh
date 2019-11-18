#!/bin/bash

VIDEOS='crossroad crossroad2 crossroad3 crossroad4 drift driving1 driving2
driving_downtown highway highway_normal_traffic nyc jp jp_hw lane_split
motorway park russia russia1 traffic tw tw1 tw_road tw_under_bridge'

# VIDEOS='motorway'

for VIDEO in $VIDEOS
do
    python awstream_spatial_overfitting.py \
        --video $VIDEO \
        --output spatial_overfitting_results_11_06/awstream_spatial_overfitting_results_${VIDEO}.csv \
        --log spatial_overfitting_profile_11_06/awstream_spatial_overfitting_profile_${VIDEO}.csv \
        --short_video_length 30 \
        --profile_length 30 &
done
