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
    --profile_filename {PathToSaveOutput}/videostorm_interface_profile_tv_show.csv \
    --output_filename {PathToSaveOutput}/videostorm_interface_results_tv_show.csv \
    --video_save_path {PathToSaveOutput} \
    --videostorm_temporal_flag 1 \
    --videostorm_spacial_flag 1 \
    --videostorm_model_flag 1
```

- videostorm interface params
1. `pipeline init`: `temporal_sampling_list, model_list, original_resolution, spacial_resolution, quantizer_list, profile_log, video_save_path, videostorm_temporal_flag, videostorm_spacial_flag, videostorm_model_flag`
2. `Source/Server`: input-`clip, pruned_video_dict, original_video, frame_range`, return-`best_frame_rate, best_model`
3. `evaluate`: input-`clip, video, original_video, best_frame_rate, best_spacial_choice, frame_range`, return-`f1_score, relative_gpu_time, relative_bandwidth`

- Pycharm Config example
```sh
--video
tv_show
--data_root
/Users/apple/Desktop/video/benchmarking/cbn
--dataset
youtube
--short_video_length
30
--profile_length
30
--overfitting
--sample_step_list
5
--original_resolution
720p
--spacial_resolution
720p
--model_list
faster_rcnn_resnet101
faster_rcnn_inception_v2
ssd_mobilenet_v2
--classes_interested
person
--coco_label_file
../mscoco_label_map.pbtxt
--profile_filename
/Users/apple/Desktop/video/benchmarking/cbn/output/compare/videostorm_interface_profile_tv_show.csv
--output_filename
/Users/apple/Desktop/video/benchmarking/cbn/output/compare/videostorm_interface_results_tv_show.csv
--video_save_path 
/Users/apple/Desktop/video/benchmarking/cbn/output/save
--videostorm_temporal_flag
1
--videostorm_spacial_flag
1
--videostorm_model_flag
1
```