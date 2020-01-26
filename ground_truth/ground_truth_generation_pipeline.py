


import os
from benchmarking.constants import babygroot_model_path, MODEL_PATH, babygroot_DT_ROOT
from create_youtube_video_input import create_tensorflow_inputrecord
from infer_detections_for_ground_truth import infer_ground_truth
from infer_object_id import infer_object_id



def gt_generation_pipeline(videoname, resol, model, gpu):
    inference_graph = os.path.join(babygroot_model_path, MODEL_PATH[model], 'frozen_inference_graph.pb') 
    data_path = os.path.join(babygroot_DT_ROOT, videoname, resol)
    tensorflow_record_path = os.path.join(data_path, 'profile')
    print(tensorflow_record_path)
    create_tensorflow_inputrecord(data_path, tensorflow_record_path, resol, extension='jpg')
    print("Done creating input!")
    # import pdb
    # pdb.set_trace()

    # start inferring ground truth
    all_input_tfrecord_paths = os.path.join(tensorflow_record_path, 'input.record')
    output_tfrecord_path = os.path.join(tensorflow_record_path, 'gt_'+ model + '_COCO.record')
    gt_csv = os.path.join(tensorflow_record_path, 'gt_'+ model + '_COCO.csv') 
    output_time_path = os.path.join(tensorflow_record_path, 'full_model_time_'+ model + '_COCO.csv')
    infer_ground_truth(all_input_tfrecord_paths, 
                        inference_graph, 
                        output_tfrecord_path, 
                        gt_csv, 
                        output_time_path, 
                        gpu)

    # smooth labels
    final_gt_file = os.path.join(tensorflow_record_path, 'updated_gt_' + model + '_COCO_no_filter.csv')
    infer_object_id(gt_csv, final_gt_file)

    return

def main():
    videoname = 'street_racing'
    model = 'FasterRCNN'
    gpu = '0'
    resol = '720p'
    gt_generation_pipeline(videoname, resol, model, gpu)
    return

if __name__ == '__main__':
    main()