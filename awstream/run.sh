#!/bin/bash

VIDEOS='crossroad crossroad2 crossroad3 crossroad4 drift driving1 driving2
driving_downtown highway highway_normal_traffic nyc jp jp_hw lane_split
motorway park russia russia1 traffic tw tw1 tw_road tw_under_bridge road_trip'

# VIDEOS='crossroad2'

SHORT_VIDEO_LENGTH=2
PROFILE_LENGTH=2

for VIDEO in $VIDEOS
do
    python awstream_spatial_overfitting.py \
        --video $VIDEO \
        --output spatial_overfitting_results_${SHORT_VIDEO_LENGTH}s/awstream_spatial_overfitting_results_${VIDEO}.csv \
        --log spatial_overfitting_profile_${SHORT_VIDEO_LENGTH}s/awstream_spatial_overfitting_profile_${VIDEO}.csv \
        --short_video_length $SHORT_VIDEO_LENGTH \
        --profile_length $PROFILE_LENGTH &
done
