```sh
cd {PathToBenchmarking}/benchmarking
python videostorm \
    --video tv_show \
    --dataset youtube \
    --data_root {PathToDataset} \
    --short_video_length 30 \
    --profile_length 10 \
    --model_list faster_rcnn_resnet101 faster_rcnn_inception_v2 ssd_mobilenet_v2 \
    --overfitting \
    --classes_interested person \
    --output_filename {PathToSaveOutput}/videostorm_results_tv_show.csv \
    --profile_filename {PathToSaveOutput}/videostorm_profile_tv_show.csv
```
