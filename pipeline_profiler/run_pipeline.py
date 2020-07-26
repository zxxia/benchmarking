import csv
import os

from utils.utils import load_COCOlabelmap
from videos import get_dataset_class, get_seg_paths
from videostorm.VideoStorm import VideoStorm

def run_pipeline(args):
    dataset_class = get_dataset_class(args.dataset)
    seg_paths = get_seg_paths(args.data_root, args.dataset, args.video)
    original_resolution = args.original_resolution
    model_list = args.model_list
    temporal_list = args.temporal_list
    spatial_list = args.spatial_list
    short_video_length = args.short_video_length
    coco_id2name, coco_name2id = load_COCOlabelmap(args.coco_label_file)
    classes_interested = {coco_name2id[class_type]
                          for class_type in args.classes_interested}
    proxy_profile_filename='tmp_profile.csv'
    # pipeline = VideoStorm(temporal_list, model_list, proxy_profile_filename)
    # pipeline = VideoStorm(temporal_list ['1', ], spatial_list ['720p'], model_list ['frnn resnet'], proxy_profile_filename, 1, 0, 1)
    # pipeline = VideoStorm(temporal_list, model_list, original_resolution, spatial_list, proxy_profile_filename, args.temporal_flag, args.spacial_flag, args.model_flag)
    
    os.remove('tmp_profile.csv')
    pipeline_performance_filename=args.pipeline_performance_filename
    with open(pipeline_performance_filename, 'w', 1) as f_out:
        writer = csv.writer(f_out) 
        writer.writerow(
            ["video_name", 'model', 'temporal', 'spatial',"f1" 'gpu time', 'bandwidth'])
        for seg_path in seg_paths:
            print(seg_path)
            seg_name = os.path.basename(seg_path)
            # loading videos
            print(dataset_class)
            original_video = dataset_class(
                seg_path, seg_name, original_resolution, 'faster_rcnn_resnet101',
                filter_flag=True, classes_interested=classes_interested)
            videos = {}
            for model in model_list:
                video = dataset_class(seg_path, seg_name, original_resolution,
                                      model, filter_flag=True,
                                      classes_interested=classes_interested)
                videos[model] = video

            # compute number of short videos can be splitted
            if original_video.duration > short_video_length:
                nb_short_videos = original_video.frame_count // (
                    short_video_length*original_video.frame_rate)
            else:
                nb_short_videos = 1

            for i in range(nb_short_videos):
                clip = seg_name + '_' + str(i)
                start_frame = i * short_video_length * \
                    original_video.frame_rate + \
                    original_video.start_frame_index

                end_frame = min((i + 1) * short_video_length *
                                original_video.frame_rate,
                                original_video.end_frame_index)
                print('{} start={} end={}'.format( \
                    clip, start_frame, end_frame))
                test_start = start_frame
                test_end = end_frame
                # print('Evaluate {} start={} end={}'.format(clip, test_start, test_end))
                
                #model pruning
                cur_temporal=args.oracle_temporal
                cur_spatial=args.spatial_temporal
                for cur_model in model_list:
                    f1_score, relative_gpu_time, relative_bandwidth = \
                        pipeline.evaluate(videos[cur_model], original_video,
                                        cur_temporal,cur_spatial, [test_start, test_end])
                    writer.writerow(
                        [clip, cur_model, cur_temporal,cur_spatial,
                         f1_score, relative_gpu_time, relative_bandwidth])
                
                #temporal pruning
                cur_model=args.oracle_model
                cur_spatial=args.oracle_spatial
                for cur_temporal in temporal_list:
                    f1_score, relative_gpu_time, relative_bandwidth = \
                        pipeline.evaluate(videos[cur_model], original_video,
                                        cur_temporal,cur_spatial, [test_start, test_end])
                    writer.writerow(
                        [clip, cur_model, cur_temporal,cur_spatial,
                         f1_score, relative_gpu_time, relative_bandwidth])

                #spatial pruning
                cur_temporal=args.oracle_temporal
                cur_model=args.oracle_model
                for cur_spatial in spatial_list:
                    f1_score, relative_gpu_time, relative_bandwidth = \
                        pipeline.evaluate(videos[cur_model], original_video,
                                        cur_temporal,cur_spatial, [test_start, test_end])
                    writer.writerow(
                        [clip, cur_model, cur_temporal,cur_spatial,
                         f1_score, relative_gpu_time, relative_bandwidth])         

                #temporal pruning 
                cur_model=args.oracle_model
                cur_spatial=args.oracle_spatial
                for cur_temporal in temporal_list:
                    f1_score, relative_gpu_time, relative_bandwidth = \
                        pipeline.evaluate(videos[cur_model], original_video,
                                        cur_temporal, cur_spatial, [test_start, test_end])
                    writer.writerow(
                        [clip, cur_model, cur_temporal,cur_spatial,
                         f1_score, relative_gpu_time, relative_bandwidth])  



                
