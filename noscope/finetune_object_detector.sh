
DATASET='driving2'
GPU="1"

RESOL='720p'
DATA_PATH='/mnt/data/zhujun/dataset/Youtube/'
OUTPUT='./'${DATASET}'/data/'

python create_youtube_tf_record.py \
     --resol=$RESOL \
     --data_path=$DATA_PATH \
     --dataset_name=$DATASET \
     --output_path=$OUTPUT \
     --train_range='1,18001' \
     --val_range='18001,22001' \



# CODE_PATH="/home/zhujunxiao/zxxia/models/research/object_detection"
PIPELINE_CONFIG_PATH="./configs/ssd_mobilenet_v2_"${DATASET}".config"
MODEL_DIR="./"${DATASET}"/trained_models/"
NUM_TRAIN_STEPS=2000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1


python ./finetune_model/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --gpu=$GPU  


INPUT_TYPE=image_tensor
TRAINED_CKPT_PREFIX=$MODEL_DIR"model.ckpt-"${NUM_TRAIN_STEPS}
EXPORT_DIR="./"${DATASET}"/frozen_model/"
python ./finetune_model/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}

MODEL='mobilenetFinetuned'

# python ${CODE_PATH}/dataset_tools/create_youtube_video_input.py \
#    --metadata_file=${DATA_PATH}${DATASET}/metadata.json \
#    --data_path=${DATA_PATH}${DATASET}/${RESOL} \
#    --output_path=${DATA_PATH}${DATASET}/${RESOL}/profile/ \
#    --resol=$RESOL

python ./ground_truth/infer_detections_for_ground_truth.py \
    --inference_graph=${EXPORT_DIR}/frozen_inference_graph.pb \
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
