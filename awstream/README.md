```sh
cd {PathToBenchmarking}/benchmarking
python awstream \
    --video tv_show \
    --dataset youtube \
    --data_root {PathToDataset} \
    --short_video_length 30 \
    --profile_length 10 \
    --output_filename {PathToSaveOutput}/awstream_results_tv_show.csv \
    --profile_filename {PathToSaveOutput}/awstream_profile_tv_show.csv \
    --classes_interested person \
    --video_save_path {PathToSaveOutput}
```
