TRAIN_DATASET='crossroad2_night'
DATASET='crossroad2'
GPU="1"

RESOL='720p'
DATA_PATH='/mnt/data/zhujun/dataset/Youtube/'
OUTPUT='/mnt/data/zhujun/dataset/NoScope_finetuned_models/'${DATASET}'/data/'
MODEL_DIR="/mnt/data/zhujun/dataset/NoScope_finetuned_models/"${TRAIN_DATASET}"/frozen_model/"

MODEL='mobilenetFinetuned_by_nighttimedata'

# python ${CODE_PATH}/dataset_tools/create_youtube_video_input.py \
#    --metadata_file=${DATA_PATH}${DATASET}/metadata.json \
#    --data_path=${DATA_PATH}${DATASET}/${RESOL} \
#    --output_path=${DATA_PATH}${DATASET}/${RESOL}/profile/ \
#    --resol=$RESOL

python ../ground_truth/infer_detections_for_ground_truth.py \
    --inference_graph=${MODEL_DIR}/frozen_inference_graph.pb \
    --discard_image_pixels \
    --gpu=$GPU \
    --input_tfrecord_paths=${DATA_PATH}${DATASET}/${RESOL}/profile/input.record \
    --output_tfrecord_path=${OUTPUT}/gt_${MODEL}_COCO.record \
    --output_time_path=${OUTPUT}/full_model_time_${MODEL}_COCO.csv \
    --gt_csv=${OUTPUT}/gt_${MODEL}_COCO.csv

python ./finetune_model/infer_finetuned_mobilenet_object_id.py \
    --resol=$RESOL \
    --input_file=${OUTPUT}/gt_${MODEL}_COCO.csv \
    --output_file=${OUTPUT}/updated_gt_${MODEL}_COCO_no_filter.csv
