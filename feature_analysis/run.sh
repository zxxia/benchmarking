#!/bin/bash

# FEATURE="'Object Area'"
# FEATURE='Object Velocity'
# FEATURE='Total Object Area'
# VIDEOS='crossroad crossroad2 crossroad3 crossroad4 drift driving1 driving2
# driving_downtown highway highway_normal_traffic nyc jp jp_hw lane_split
# motorway park russia russia1 traffic tw tw1 tw_road tw_under_bridge'
VIDEOS="road_trip"
# rm results_10_17.csv
for VIDEO in ${VIDEOS}
do
python feature_scanning.py \
    --video ${VIDEO} \
    --metadata /data/zxxia/videos/${VIDEO}/metadata.json \
    --feature_file /data/zxxia/benchmarking/results/videos/${VIDEO}/720p/profile/Video_features_${VIDEO}_object_width_height_type_filter.csv \
    --feature_file_simple /data/zxxia/benchmarking/results/videos/${VIDEO}/720p/profile/Video_features_${VIDEO}_object_width_height_type_filter_mobilenet.csv \
    --output_file results_30s_100.csv \
    --short_video_length 30
done

# for VIDEO in ${VIDEOS}
# do
# python feature_scanning.py \
#     --video ${VIDEO} \
#     --metadata /data2/zxxia/videos/${VIDEO}/metadata.json \
#     --feature_file /data/zxxia/benchmarking/results/videos/${VIDEO}/720p/profile/Video_features_${VIDEO}_object_width_height_type_filter.csv \
#     --output_file results_30s_100.csv \
#     --short_video_length 30
# done
