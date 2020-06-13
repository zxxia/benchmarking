```sh
cd {PathToBenchmarking}/benchmarking
python vigil \
    --video tv_show \
    --dataset youtube \
    --data_root {PathToDataset} \
    --short_video_length 30 \
    --profile_length 10 \
    --output_filename {PathToSaveOutput}/vigil_results.csv \
    --simple_model faster_rcnn_inception_v2 \
    --classes_interested person \
    --video_save_path {PathToSaveOutput}
```
