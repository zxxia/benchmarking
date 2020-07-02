```sh
cd {PathToBenchmarking}/benchmarking
python feature_scanner \
    --video tv_show \
    --dataset youtube \
    --data_root {PathToDataset} \
    --short_video_length 30 \
    --granularity high \
    --output_filename {PathToSaveOutput}/features.csv \
    --classes_interested person \
```
