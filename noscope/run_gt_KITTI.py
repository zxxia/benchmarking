import glob
import os
import subprocess

TRAIN_DATASET='KITTI'
DATASET='KITTI'
GPU="1"

RESOL='720p'
DATA_PATH='/mnt/data/zhujun/dataset/KITTI'
OUTPUT='/mnt/data/zhujun/dataset/NoScope_finetuned_models/' + DATASET + '/data/'
MODEL_DIR="/mnt/data/zhujun/dataset/NoScope_finetuned_models/" + TRAIN_DATASET + "/frozen_model/frozen_inference_graph.pb"

MODEL='mobilenetFinetuned'

def main():
    # for img_path in sorted(glob.glob(os.path.join(ROOT, '*/*/img1'))):
    for resol in ['720p']:
        # for img_path in sorted(glob.glob(os.path.join(ROOT, '*/*/img1'))):
        for img_path in sorted(glob.glob(os.path.join(DATA_PATH, '*/*/image_02/data'))):
            video_name = img_path.split('/')[-4] + '_' +  img_path.split('/')[-3].split('_')[-2]
            print(video_name)
            output_path = os.path.join(OUTPUT, video_name)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            cmd = ["python", "../ground_truth/create_youtube_video_input.py",
                   "--data_path="+img_path+'/'+resol, 
                   "--output_path="+output_path,
                   '--resol='+resol]

            print(cmd)
            subprocess.run(cmd, check=True)
            cmd = ['python', '../ground_truth/infer_detections_for_ground_truth.py',
                   '--inference_graph=' + MODEL_DIR,
                   '--discard_image_pixels',
                   '--gpu=2',
                   '--input_tfrecord_paths='+
                   os.path.join(output_path, 'input.record'),
                   '--output_tfrecord_path=' +
                   os.path.join(output_path, 'gt_'+MODEL+'_COCO.record'),
                   '--output_time_path='+os.path.join(
                       output_path, 'full_model_time_'+MODEL+'_COCO.csv'),
                   '--gt_csv='+os.path.join(output_path, 'gt_'+MODEL+'_COCO.csv')]
            print(cmd)
            subprocess.run(cmd, check=True)
            cmd = ['python', './finetune_model/infer_finetuned_mobilenet_object_id.py',
                   '--resol='+resol,
                   '--input_file='+
                   os.path.join(output_path, 'gt_'+MODEL+'_COCO.csv'),
                   '--output_file='+
                   os.path.join(output_path, 'updated_gt_'+MODEL+'_COCO_no_filter.csv')]
            print(cmd)
            subprocess.run(cmd, check=True)



if __name__ == '__main__':
    main()
