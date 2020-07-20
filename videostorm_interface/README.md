```sh
cd {PathToBenchmarking}/benchmarking
python videostorm_interface \
    --video tv_show \
    --data_root {PathToDataset} \
    --dataset youtube \
    --short_video_length 30 \
    --profile_length 30 \
    --overfitting \
    --sample_step_list 5 \
    --original_resolution 720p \
    --spacial_resolution 360p \
    --model_list faster_rcnn_resnet101 faster_rcnn_inception_v2 ssd_mobilenet_v2 \
    --classes_interested car truck person \
    --coco_label_file ../mscoco_label_map.pbtxt \
    --profile_filename {PathToSaveOutput}/videostorm_profile_tv_show.csv \
    --output_filename {PathToSaveOutput}/videostorm_results_tv_show.csv \
    --videostorm_temporal_flag 1 \
    --videostorm_spacial_flag 1 \
    --videostorm_model_flag 1
```