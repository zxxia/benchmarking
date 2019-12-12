#!/bin/bash

VIDEOS='crossroad crossroad2 crossroad3 crossroad4 drift driving1 driving2
driving_downtown highway highway_normal_traffic nyc jp jp_hw lane_split
motorway park russia russia1 traffic tw tw1 tw_road tw_under_bridge'

# VIDEOS='road_trip'
# VIDEOS='traffic'
#
for VIDEO in $VIDEOS
do
    python videostorm_overfitting.py \
        --video $VIDEO \
        --input /data/zxxia/benchmarking/results/videos/${VIDEO}/720p/profile/updated_gt_FasterRCNN_COCO_no_filter.csv \
        --metadata_file /data/zxxia/videos/${VIDEO}/metadata.json \
        --output test_coverage_results/videostorm_coverage_results_${VIDEO}.csv \
        --log test_coverage_profile/videostorm_coverage_profile_${VIDEO}.csv \
        --short_video_length 30 \
        --profile_length 10 &
done



# For KITTI dataset

# DATA_PATH='/data2/zxxia/KITTI/'
# RESULT_PATH='/data/zxxia/benchmarking/results/KITTI/'
# LOCATIONS="City Residential Road"
# for LOCATION in ${LOCATIONS}
# do
#     SEGMENTS=$(cd ${DATA_PATH}/${LOCATION} && ls -d */)
#
#     for SEGMENT in ${SEGMENTS}
#     do
#         # mkdir -p ${RESULT_PATH}/${LOCATION}/${SEGMENT}/375p/profile
#         # ls $SEGMENT/image_02/data/375p/*.png
#         let end=$(ls -l ${DATA_PATH}/${LOCATION}/${SEGMENT}/image_02/data/375p/*.png | wc -l)
#         let start=0
#         let end--
#         let duration="($end/10)+1"
#         VIDEO=kitti_${LOCATION}_${SEGMENT//\/}
#         let profile_len=$duration/3
#         if (( duration > 20)); then
#             echo ${VIDEO}
#             echo $start $end $duration
#             python videostorm_overfitting.py \
#                 --video ${VIDEO} \
#                 --input ${RESULT_PATH}${LOCATION}/${SEGMENT}/375p/profile/updated_gt_FasterRCNN_COCO_no_filter.csv \
#                 --output kitti_coverage_results/videostorm_coverage_results_${VIDEO}.csv \
#                 --log kitti_coverage_profile/videostorm_coverage_profile_${VIDEO}.csv \
#                 --short_video_length $duration \
#                 --profile_length $profile_len \
#                 --frame_rate 10
#         fi
#     done
# done

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
