```sh
cd {PathToBenchmarking}/benchmarking
python awstream \
    --video tv_show \
    --data_root {PathToDataset} \
    --dataset youtube \
    --short_video_length 30 \
    --profile_length 10 \
    --output_filename {PathToSaveOutput}/awstream_results_tv_show.csv \
    --profile_filename {PathToSaveOutput}/awstream_profile_tv_show.csv \
    --classes_interested person \
    --video_save_path {PathToSaveOutput} \
    --awstream_temporal_flag 1 \
    --awstream_spacial_flag 1 \
```

- videostorm interface params
1. `pipeline init`: `temporal_sampling_list, model_list, original_resolution, spacial_resolution_list, quantizer_list, profile_log, video_save_path, awstream_temporal_flag, awstream_spacial_flag, awstream_model_flag`
2. `Source/Server`: input-`clip, pruned_video_dict, original_video, frame_range`, return-`best_resol, best_fps, best_relative_bw`
3. `evaluate`: input-`clip, video, original_video, best_frame_rate, best_spacial_choice, frame_range`, return-`f1_score, relative_gpu_time, relative_bandwidth`


- Pycharm Config Example
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
10
--sample_step_list
5
--original_resolution
720p
--classes_interested 
person
--coco_label_file
../mscoco_label_map.pbtxt
--output_filename 
/Users/apple/Desktop/video/benchmarking/cbn/output/compare/awstream_results_tv_show.csv
--profile_filename
/Users/apple/Desktop/video/benchmarking/cbn/output/compare/awstream_profile_tv_show.csv
--video_save_path 
/Users/apple/Desktop/video/benchmarking/cbn/output/save
--awstream_temporal_flag
1
--awstream_spacial_flag
1
```