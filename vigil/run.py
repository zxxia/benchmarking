"""Vigil offline simulation driver."""
import csv
import os

from vigil.Vigil import Vigil, mask_video_ffmpeg
from utils.utils import load_COCOlabelmap
from videos import get_dataset_class, get_seg_paths


def run(args):
    """Run Vigil simulation."""
    dataset_class = get_dataset_class(args.dataset)
    seg_paths = get_seg_paths(args.data_root, args.dataset, args.video)
    original_resolution = args.original_resolution
    short_video_length = args.short_video_length
    profile_length = args.profile_length
    output_filename = args.output_filename
    crop_flag = args.crop
    overfitting = args.overfitting
    # detection_path = args.detection_root
    output_path = args.video_save_path
    simple_model = args.simple_model

    coco_id2name, coco_name2id = load_COCOlabelmap(args.coco_label_file)
    classes_interested = {coco_name2id[class_type]
                          for class_type in args.classes_interested}

    if crop_flag:
        # TODO: do cropping
        # assert short_video_length == profile_length, "short_video_length " \
        #     "should equal to profile_length when overfitting."
        for seg_path in seg_paths:
            print(seg_path)
            seg_name = os.path.basename(seg_path)
            # loading videos
            video = dataset_class(seg_path, seg_name, original_resolution,
                                  simple_model, filter_flag=True,
                                  classes_interested=classes_interested)
            cropped_img_path = video.image_path + "_cropped"
            print(f'Cropping into {cropped_img_path}...')
            if not os.path.exists(cropped_img_path):
                os.mkdir(cropped_img_path)
            mask_video_ffmpeg(video, 0.1, 0.1, save_path=cropped_img_path)
        return

    pipeline = Vigil()
    with open(output_filename, 'w', 1) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['video', 'mp4 bw', 'f1', 'jpg bw', 'original bw'])
        for seg_path in seg_paths:
            print(seg_path)
            seg_name = os.path.basename(seg_path)
            # loading videos
            original_video = dataset_class(
                seg_path, seg_name, original_resolution,
                'faster_rcnn_resnet101', filter_flag=True,
                classes_interested=classes_interested)
            video = dataset_class(
                seg_path, seg_name, original_resolution,
                'faster_rcnn_resnet101', filter_flag=True,
                classes_interested=classes_interested, cropped=True)

            # compute number of short videos can be splitted
            if video.duration > short_video_length:
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

                if overfitting:
                    test_start = start_frame
                    test_end = end_frame
                else:
                    test_start = profile_end + 1
                    test_end = end_frame

                print('Evaluate {} start={} end={}'.format(
                    clip, test_start, test_end))
                jpg_bw, f1, mp4_bw, original_bw = pipeline.evaluate(
                    clip, video, original_video, [test_start, test_end],
                    output_path)
                writer.writerow([clip, mp4_bw, f1, jpg_bw, original_bw])
