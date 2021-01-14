import csv
import os

from reducto.Reducto import Reducto
from utils.utils import load_COCOlabelmap  # , write_json_file
from videos import get_dataset_class, get_seg_paths


def run(args):
    print(args)
    dataset_class = get_dataset_class(args.dataset)
    seg_paths = get_seg_paths(args.data_root, args.dataset, args.video)
    original_resolution = args.original_resolution
    overfitting = args.overfitting
    short_video_length = args.short_video_length
    profile_length = args.profile_length
    output_filename = args.output_filename
    profile_filename = args.profile_filename

    coco_id2name, coco_name2id = load_COCOlabelmap(args.coco_label_file)
    classes_interested = {coco_name2id[class_type]
                          for class_type in args.classes_interested}

    if overfitting:
        assert short_video_length == profile_length, "short_video_length " \
            "should equal to profile_length when overfitting."
    else:
        assert short_video_length >= profile_length, "short_video_length " \
            "should no less than profile_length."
    with open(profile_filename, 'w', 1) as f_profile, \
            open(output_filename, 'w', 1) as f_out:
        profile_writer = csv.writer(f_profile, lineterminator='\n')
        profile_writer.writerow(['video', 'feature_type', 'feature_threshold',
                                 'frame_rate', 'f1'])
        writer = csv.writer(f_out, lineterminator='\n')
        writer.writerow(['video', 'feature_type', 'feature_threshold',
                         'frame_rate', 'bandwidth', 'f1'])
        for seg_path in seg_paths:
            seg_name = os.path.basename(seg_path)
            original_video = dataset_class(
                seg_path, seg_name, original_resolution,
                'faster_rcnn_resnet101',
                filter_flag=True, classes_interested=classes_interested)
            # write_json_file('{}.json'.format(original_video.video_name),
            #                 generate_feature_thresholds(original_video, [1,
            #                 100]))
            # compute number of short videos can be splitted
            if original_video.duration > short_video_length:
                nb_short_videos = original_video.frame_count // (
                    short_video_length*original_video.frame_rate)
            else:
                nb_short_videos = 1
            pipeline = Reducto(original_video)
            # best_feat_type, best_thresh, thresholds = pipeline.profile(
            #     (1, 600))
            # f1_scores, perfs = pipeline.evaluate(best_feat_type, (601, 18000))
            for i in range(nb_short_videos):
                clip = seg_name + '_' + str(i)
                start_frame = i * short_video_length * \
                    original_video.frame_rate + \
                    original_video.start_frame_index

                end_frame = min((i + 1) * short_video_length *
                                original_video.frame_rate,
                                original_video.end_frame_index)
                print('{} start={} end={}'.format(
                    clip, start_frame, end_frame))
                # use 30 seconds video for profiling
                if overfitting:
                    profile_start = start_frame
                    profile_end = end_frame
                else:
                    profile_start = start_frame
                    profile_end = start_frame + original_video.frame_rate * \
                        profile_length - 1

                print('profile {} start={} end={}'.format(
                    clip, profile_start, profile_end))
                best_feat_type, best_thresh, profile_results = \
                    pipeline.profile_segment((profile_start, profile_end))
                for feat_type, feat_results in profile_results.items():
                    for thresh, results in feat_results.items():
                        profile_writer.writerow(
                            [clip, feat_type, thresh, results['frame_rate'],
                             results['f1']])

                # print("Profile {}: best resol={}, best fps={}, best bw={}"
                #       .format(clip, best_feat_type, best_thresh, best_bw))

                test_start = start_frame
                test_end = end_frame

                print('Evaluate {} start={} end={}'.format(
                    clip, test_start, test_end))
                f1_score, percent_frames_sent, relative_bw = \
                    pipeline.evaluate_segment(
                        clip, best_feat_type, best_thresh,
                        (test_start, test_end), args.save_dir)

                # print('{} best fps={}, best resolution={} ==> tested f1={}'
                #       .format(clip, best_fps/original_video.frame_rate,
                #               best_resol, f1_score))
                writer.writerow([clip, best_feat_type, best_thresh,
                                 percent_frames_sent, relative_bw, f1_score])
