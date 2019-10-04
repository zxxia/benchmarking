#!/bin/bash

# overlap_percent='0.1 0.2 0.3 0.4 0.5 0.6'
# vote_percent='0.1 0.2 0.3 0.4 0.5 0.6'
DATA_PATH=/mnt/data/zhujun/dataset/Youtube
VIDEOS="crossroad2 crossroad3 crossroad4 drift driving_downtown
        highway jp lane_split motorway nyc park russia russia1 traffic tw tw1
        tw_under_bridge driving1 driving2 highway_normal_traffic"
# VIDEOS='crossroad'
for dataset in $VIDEOS
do
    echo $VIDEOS
    input_file=${DATA_PATH}/${dataset}/720p/profile/gt_FasterRCNN_COCO.csv
    if [ ! -f ${input_file} ]; then
        echo ${input_file}" not found!"
        input_file=${DATA_PATH}/${dataset}/720p/profile/gt_FasterRCNN_COCO_720p.csv
    fi
    python infer_object_id.py \
        --metadata_file=${DATA_PATH}/${dataset}/metadata.json \
        --input_file=${input_file} \
        --output_file=${DATA_PATH}/${dataset}/720p/profile/Parsed_gt_FasterRCNN_COCO_filtered.csv \
        --updated_gt_file=${DATA_PATH}/${dataset}/720p/profile/updated_gt_FasterRCNN_COCO_filtered.csv
    echo "Done infer object id"
    python video_feature_youtube.py \
        --metadata_file=${DATA_PATH}/${dataset}/metadata.json \
        --input_file=${DATA_PATH}/${dataset}/720p/profile/Parsed_gt_FasterRCNN_COCO_filtered.csv \
        --output_file=${DATA_PATH}/${dataset}/720p/profile/Video_features_${dataset}_filtered.csv
done
