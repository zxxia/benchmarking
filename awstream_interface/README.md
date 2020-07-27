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
    --video_save_path {PathToSaveOutput}
    --awstream_temporal_flag 1 \
    --awstream_spacial_flag 1 \
```

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