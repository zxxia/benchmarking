#!/bin/bash
# VIDEO=jp
# VIDEO=russia1
# VIDEO=highway
# VIDEO=driving_downtown
VIDEO=road_trip

RESOL1=720p
RESOL2=360p
python show_annot.py \
    --video ${VIDEO} \
    --detection_file1 /data/zxxia/benchmarking/results/videos/${VIDEO}/${RESOL1}/profile/updated_gt_FasterRCNN_COCO_no_filter.csv \
    --resol1 ${RESOL1} \
    --image_path /data2/zxxia/videos/${VIDEO}/${RESOL1}/ \
    --detection_file2 /data/zxxia/benchmarking/results/videos/${VIDEO}/${RESOL2}/profile/updated_gt_FasterRCNN_COCO_no_filter.csv \
    --resol2 ${RESOL2} \
    --start_frame 28623 \
    --end_frame 500000 \
    --visualize \
    --output_folder ${VIDEO}_vis_area
    #--save
