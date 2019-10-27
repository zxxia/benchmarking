#!/bin/bash

# VIDEOS='crossroad crossroad2 crossroad3 crossroad4 drift driving1 driving2
# driving_downtown highway highway_normal_traffic nyc jp jp_hw lane_split
# motorway park russia russia1 traffic tw tw1 tw_road tw_under_bridge'

VIDEOS='road_trip'


# for VIDEO in $VIDEOS
# do
#     python videostorm_overfitting.py \
#         --video $VIDEO \
#         --input /data/zxxia/benchmarking/results/videos/${VIDEO}/720p/profile/updated_gt_FasterRCNN_COCO_no_filter.csv \
#         --metadata_file /data2/zxxia/videos/${VIDEO}/metadata.json \
#         --output overfitting_results_10_17_10s/videostorm_overfitting_results_${VIDEO}.csv \
#         --log overfitting_profile_10_17_10s/videostorm_overfitting_profile_${VIDEO}.csv \
#         --short_video_length 10 \
#         --profile_length 10
# done

for VIDEO in $VIDEOS
do
    python videostorm_overfitting.py \
        --video $VIDEO \
        --input /data/zxxia/benchmarking/results/videos/${VIDEO}/720p/profile/updated_gt_FasterRCNN_COCO_no_filter.csv \
        --metadata_file /data/zxxia/videos/${VIDEO}/metadata.json \
        --output overfitting_results_10_21_30s/videostorm_overfitting_results_${VIDEO}.csv \
        --log overfitting_profile_10_21_30s/videostorm_overfitting_profile_${VIDEO}.csv \
        --short_video_length 30 \
        --profile_length 30
done

# for VIDEO in $VIDEOS
# do
#     python videostorm_baseline.py \
#         --video $VIDEO \
#         --input /data/zxxia/benchmarking/results/videos/${VIDEO}/720p/profile/updated_gt_FasterRCNN_COCO_no_filter.csv \
#         --output baseline_results_10_17_10s_50/videostorm_baseline_results_${VIDEO}.csv \
#         --metadata_file /data/zxxia/videos/${VIDEO}/metadata.json \
#         --log baseline_profile_10_17_10s_50/videostorm_baseline_profile_${VIDEO}.csv \
#         --sample_rate 50 \
#         --short_video_length 10 \
#         --profile_length 10 &
# done
# for VIDEO in $VIDEOS
# do
#     python videostorm_baseline_v2.py \
#         --video $VIDEO \
#         --input /data/zxxia/benchmarking/results/videos/${VIDEO}/720p/profile/updated_gt_FasterRCNN_COCO_no_filter.csv \
#         --output baseline_results_30s_20/videostorm_baseline_results_${VIDEO}.csv \
#         --metadata_file /data2/zxxia/videos/${VIDEO}/metadata.json \
#         --log baseline_profile_30s_20/videostorm_baseline_profile_${VIDEO}.csv \
#         --sample_rate 20 \
#         --short_video_length 30 \
#         --profile_length 30
# done
