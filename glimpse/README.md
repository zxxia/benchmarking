```sh
cd {PathToBenchmarking}/benchmarking
python glimpse \
    --video tv_show \
    --dataset youtube \
    --data_root {PathToDataset} \
    --short_video_length 30 \
    --profile_length 10 \
    --frame_difference_threshold_divisor_list 20 15 10 8 5 2 \
    --output_filename glimpse_results_tv_show.csv \
    --profile_filename glimpse_profile_tv_show.csv
```
