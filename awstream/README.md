```bash
VIDEO=tv_show
SHORT_VIDEO_LENGTH=30
PROFILE_LENGTH=10
OUTPUT=person_videos
DATASET_TYPE=youtube
DATASET_ROOT=/data/zxxia/videos

python awstream \
    --video $VIDEO \
    --dataset ${DATASET_TYPE} \
    --data_root ${DATASET_ROOT}\
    --short_video_length $SHORT_VIDEO_LENGTH \
    --profile_length $PROFILE_LENGTH \
    --output_filename ${OUTPUT}/awstream_results_${VIDEO}.csv \
    --profile_filename ${OUTPUT}/awstream_profile_${VIDEO}.csv \
    --classes_interested person \
    --video_save_path ${OUTPUT}
```
