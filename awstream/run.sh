#!/bin/bash

VIDEOS='crossroad crossroad2 crossroad3 crossroad4 drift driving1 driving2
driving_downtown highway highway_normal_traffic nyc jp jp_hw lane_split
motorway park russia russia1 traffic tw tw1 tw_road tw_under_bridge road_trip'

# VIDEOS='road_trip'

SHORT_VIDEO_LENGTH=30
PROFILE_LENGTH=10

for VIDEO in $VIDEOS
do
    python awstream_e2e.py \
        --video $VIDEO \
        --output awstream_e2e_${VIDEO}.csv \
        --log awstream_e2e_profile_${VIDEO}.csv \
        --short_video_length $SHORT_VIDEO_LENGTH \
        --profile_length $PROFILE_LENGTH &
        --video_save_path .
done
# for VIDEO in $VIDEOS
# do
#     python awstream_spatial_overfitting.py \
#         --video $VIDEO \
#         --output ./overfitting_results_30s_10s/awstream_spatial_overfitting_results_${VIDEO}.csv \
#         --log ./overfitting_results_30s_10s/awstream_spatial_overfitting_profile_${VIDEO}.csv \
#         --short_video_length $SHORT_VIDEO_LENGTH \
#         --profile_length $PROFILE_LENGTH
# done
