#!/bin/bash

VIDEOS='crossroad crossroad2 crossroad3 crossroad4 drift driving1 driving2
driving_downtown highway highway_normal_traffic nyc jp jp_hw lane_split
motorway park russia russia1 traffic tw tw1 tw_road tw_under_bridge'


for VIDEO in $VIDEOS
do
    python videostorm_overfitting.py \
        --video $VIDEO \
        --input /data/zxxia/benchmarking/results/videos/${VIDEO}/720p/profile/updated_gt_FasterRCNN_COCO_no_filter.csv \
        --output overfitting_results_10_14/videostorm_overfitting_results_${VIDEO}.csv \
        --log overfitting_profile_10_14/videostorm_overfitting_profile_${VIDEO}.csv &
done
