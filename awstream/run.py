"""AWStream offline simulation driver."""
import csv
import os

from awstream.Awstream import Awstream
from utils.utils import load_COCOlabelmap
from videos import get_dataset_class, get_seg_paths


def run(args):
    """Run AWStream simulation."""
    dataset_class = get_dataset_class(args.dataset)
    seg_paths = get_seg_paths(args.data_root, args.dataset, args.video)
    original_resolution = args.original_resolution
    resolution_list = args.resolution_list
    overfitting = args.overfitting
    sample_step_list = args.sample_step_list if not overfitting else [1]
    qp_list = args.quality_parameter_list
    short_video_length = args.short_video_length
    profile_length = args.profile_length
    profile_filename = args.profile_filename
    output_filename = args.output_filename
    # detection_path = args.detection_root
    output_path = args.video_save_path
    coco_id2name, coco_name2id = load_COCOlabelmap(args.coco_label_file)
    classes_interested = {coco_name2id[class_type]
                          for class_type in args.classes_interested}

    if overfitting:
        assert short_video_length == profile_length, "short_video_length " \
            "should equal to profile_length when overfitting."
    else:
        assert short_video_length >= profile_length, "short_video_length " \
            "should no less than profile_length."

    pipeline = Awstream(sample_step_list, resolution_list, qp_list,
                        profile_filename, output_path)
    with open(output_filename, 'w', 1) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['dataset', 'best_resolution',
                         'f1', 'frame_rate', 'bandwidth'])
        for seg_path in seg_paths:
            print(seg_path)
            seg_name = os.path.basename(seg_path)
            # loading videos
            original_video = dataset_class(
                seg_path, seg_name, original_resolution,
                'faster_rcnn_resnet101', filter_flag=True,
                classes_interested=classes_interested)
            videos = {}
            for resol in resolution_list:
                video = dataset_class(seg_path, seg_name, resol,
                                      'faster_rcnn_resnet101',
                                      filter_flag=True,
                                      classes_interested=classes_interested)
                videos[resol] = video

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
                best_resol, best_fps, best_bw = \
                    pipeline.profile(clip, videos, original_video,
                                     [profile_start, profile_end])

                print("Profile {}: best resol={}, best fps={}, best bw={}"
                      .format(clip, best_resol, best_fps, best_bw))

                if overfitting:
                    test_start = start_frame
                    test_end = end_frame
                else:
                    test_start = profile_end + 1
                    test_end = end_frame

                print('Evaluate {} start={} end={}'.format(
                    clip, test_start, test_end))
                f1_score, relative_bw = pipeline.evaluate(
                    clip, original_video, videos[str(best_resol[1])+'p'],
                    best_fps, [test_start, test_end])

                print('{} best fps={}, best resolution={} ==> tested f1={}'
                      .format(clip, best_fps/original_video.frame_rate,
                              best_resol, f1_score))
                writer.writerow([clip, str(best_resol[1]) + 'p', f1_score,
                                 best_fps/original_video.frame_rate,
                                 relative_bw])
