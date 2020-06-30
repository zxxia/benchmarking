"""NoScope offline simulation driver."""
import csv
import os

from utils.utils import load_COCOlabelmap
from videos import get_dataset_class, get_seg_paths
from noscope.NoScope import NoScope


def run(args):
    """Run NoScope simulation."""
    dataset_class = get_dataset_class(args.dataset)
    seg_paths = get_seg_paths(args.data_root, args.dataset, args.video)
    original_resolution = args.original_resolution
    overfitting = args.overfitting
    model_list = args.model_list if not overfitting else [
        'faster_rcnn_resnet101']
    sample_step_list = args.sample_step_list
    short_video_length = args.short_video_length
    profile_length = args.profile_length
    # detection_path = args.detection_root
    profile_filename = args.profile_filename
    output_filename = args.output_filename
    coco_id2name, coco_name2id = load_COCOlabelmap(args.coco_label_file)
    classes_interested = {coco_name2id[class_type]
                          for class_type in args.classes_interested}
    print(args, seg_paths)

    if overfitting:
        assert short_video_length == profile_length, "short_video_length " \
            "should equal to profile_length when overfitting."
    else:
        assert short_video_length >= profile_length, "short_video_length " \
            "should no less than profile_length."

    pipeline = NoScope(sample_step_list, model_list, profile_filename)
    with open(output_filename, 'w', 1) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            ["video_name", 'model', 'gpu time', 'frame rate', "f1"])
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
                best_frame_rate, best_model = pipeline.profile(
                    clip, videos, original_video, [profile_start, profile_end])
                best_sample_rate = original_video.frame_rate/best_frame_rate

                if overfitting:
                    test_start = start_frame
                    test_end = end_frame
                else:
                    test_start = profile_end + 1
                    test_end = end_frame

                print('Evaluate {} start={} end={}'.format(
                    clip, test_start, test_end))
                f1_score, relative_gpu_time, triggered_frames_tmp = \
                    pipeline.evaluate(videos[best_model], original_video,
                                      best_sample_rate, [test_start, test_end])

                print(clip, best_model, relative_gpu_time,
                      best_frame_rate / original_video.frame_rate, f1_score)
                writer.writerow(
                    [clip, best_model, relative_gpu_time,
                     best_frame_rate / original_video.frame_rate, f1_score])
