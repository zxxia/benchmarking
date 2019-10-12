from collections import defaultdict
from utils.model_utils import eval_single_image
from utils.utils import interpolation, compute_f1
import os
import subprocess
import copy
import pdb
from constants import RESOL_DICT



class VideoConfig:
    def __init__(self, resolution, fps, quantizer=25):
        self.resolution = resolution
        self.fps = fps
        self.quantizer = quantizer

    def debug_print(self):
        print('resolution={}, fps={}, quantizer={}'
              .format(self.resolution, self.fps, self.quantizer))


def resol_str_to_int(resol_str):
    return int(resol_str.strip('p'))


def scale(box, in_resol, out_resol):
    """
    box: [x, y, w, h]
    in_resl: (width, height)
    out_resl: (width, height)
    """
    assert(len(box) >= 4)
    ret_box = box.copy()
    x_scale = out_resol[0]/in_resol[0]
    y_scale = out_resol[1]/in_resol[1]
    ret_box[0] = int(box[0] * x_scale)
    ret_box[1] = int(box[1] * y_scale)
    ret_box[2] = int(box[2] * x_scale)
    ret_box[3] = int(box[3] * y_scale)
    return ret_box


def compress_images_to_video(list_file: str, frame_rate: str, resolution: str,
                             quality: int, output_name: str):
    ''' Compress a set of frames to a video.  '''
    cmd = ['ffmpeg', '-y', '-loglevel', 'panic', '-r', str(frame_rate), '-f',
           'concat', '-safe', '0', '-i', list_file, '-s', str(resolution),
           '-vcodec', 'libx264', '-crf', str(quality), '-pix_fmt', 'yuv420p',
           '-hide_banner', output_name]
    subprocess.run(cmd)


def compute_video_size(video, img_path, start, end, original_config, target_config):
    image_resolution = target_config.resolution
    # Create a tmp list file contains all the selected iamges
    tmp_list_file = video+'_list.txt'
    sample_rate = original_config.fps/target_config.fps
    with open(tmp_list_file, 'w') as f:
        for img_index in range(start, end + 1):
            # based on sample rate, decide whether this frame is sampled
            if img_index % sample_rate >= 1:
                continue
            else:
                line = 'file \'{}/{:06}.jpg\'\n'.format(img_path, img_index)
                f.write(line)

    frame_size = str(image_resolution[0]) + 'x' + str(image_resolution[1])
    output_video = video+'.mp4'
    compress_images_to_video(tmp_list_file, target_config.fps, frame_size,
                             target_config.quantizer, output_video)
    video_size = os.path.getsize(output_video)
    # pdb.set_trace()
    os.remove(output_video)
    os.remove(tmp_list_file)
    print('target frame rate={}, target image resolution={}, video size={}'
          .format(target_config.fps, image_resolution, video_size))
    return video_size


def profile(video, gt, dt_dict, original_config, start, end, f_profile,
            resolution_list, temporal_sampling_list, target_f1=0.9):
    result = []
    # choose resolution
    # choose frame rate

    for resolution in resolution_list:
        F1_score_list = []

        if resolution not in dt_dict:
            continue
        print('profile start={}, end={}, resolution={}, orginal resolution={}'
              .format(start, end, resolution, original_config.resolution))

        # gt = gt_dict[resolution]
        full_model_dt = copy.deepcopy(dt_dict[resolution])

        for sample_rate in temporal_sampling_list:
            tp = defaultdict(int)
            fp = defaultdict(int)
            fn = defaultdict(int)
            save_dt = []
            for img_index in range(start, end+1):
                dt_boxes_final = []
                if img_index not in full_model_dt or img_index not in gt:
                    continue
                current_full_model_dt = copy.deepcopy(full_model_dt[img_index])
                current_gt = copy.deepcopy(gt[img_index])

                # based on sample rate, decide whether this frame is sampled
                if img_index % sample_rate >= 1:
                    # this frame is not sampled, so reuse the last saved
                    # detection result
                    dt_boxes_final = [box for box in save_dt]
                else:
                    # this frame is sampled, so use the full model result
                    dt_boxes_final = [box for box in current_full_model_dt]
                    save_dt = [box for box in dt_boxes_final]

                # scale gt boxes
                # pdb.set_trace()
                for idx, box in enumerate(current_gt):
                    current_gt[idx] = scale(current_gt[idx],
                                            original_config.resolution,
                                            RESOL_DICT[resolution])
                tp[img_index], fp[img_index], fn[img_index] = \
                    eval_single_image(current_gt, dt_boxes_final)

                print(img_index, tp[img_index], fp[img_index],fn[img_index])
                # pdb.set_trace()
            tp_total = sum(tp.values())
            fp_total = sum(fp.values())
            fn_total = sum(fn.values())

            f1 = compute_f1(tp_total, fp_total, fn_total)

            f_profile.write(','.join([video, resolution, str(sample_rate),
                                      str(f1), str(tp_total), str(fp_total),
                                      str(fn_total)])+'\n')
            print('profile on {} [{}, {}], resolution={},sample rate={}, f1={}'
                  .format(video, start, end, resolution, sample_rate, f1))
            F1_score_list.append(f1)

        frame_rate_list = [original_config.fps/x for x in temporal_sampling_list]
        current_f1_list = F1_score_list

        if current_f1_list[-1] < target_f1:
            target_fps = None
        else:
            index = next(x[0] for x in enumerate(current_f1_list)
                         if x[1] > target_f1)
            if index == 0:
                target_fps = frame_rate_list[0]
            else:
                point_a = (current_f1_list[index-1], frame_rate_list[index-1])
                point_b = (current_f1_list[index], frame_rate_list[index])
                target_fps = interpolation(point_a, point_b, target_f1)
        print("Resolution = {} and target frame rate = {}"
              .format(resolution, target_fps))

        if target_fps is not None:
            tmp = copy.deepcopy(original_config)
            tmp.fps = target_fps
            tmp.resolution = RESOL_DICT[resolution]
            # TODO: tmp.quantizer = quant
            result.append(tmp)
    return result


def select_best_config(video, img_path_dict, original_config, configs, start, end):
    # select best profile
    resol = str(original_config.resolution[1]) + 'p'
    best_config = original_config
    original_bw = compute_video_size(video, img_path_dict[resol], start, end,
                                     original_config, original_config)
    min_bw = original_bw

    for config in configs:
        target_fps = config.fps

        if target_fps is None:
            continue
        # video_size = config.resolution[1]  # resol_str_to_int(resolution)
        resol = str(config.resolution[1]) + 'p'
        bw = compute_video_size(video, img_path_dict[resol], start, end,
                                original_config, config)

        if min_bw is None:
            min_bw = bw

        elif bw <= min_bw:
            best_config = config
            min_bw = bw

    return best_config, min_bw/original_bw


def profile_eval(video, img_path_dict, gt, dt, original_config, best_config,
                 start, end):
    original_resol = str(original_config.resolution[1]) + 'p'
    best_resol = str(best_config.resolution[1]) + 'p'

    origin_bw = compute_video_size(video, img_path_dict[original_resol], start, end,
                                   original_config, original_config)
    bw = compute_video_size(video, img_path_dict[best_resol], start, end,
                            original_config, best_config)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    save_dt = []
    best_sample_rate = original_config.fps / best_config.fps

    for img_index in range(start, end+1):
        if img_index not in gt or img_index not in dt:
            continue
        dt_boxes_final = []
        current_full_model_dt = copy.deepcopy(dt[img_index])
        current_gt = copy.deepcopy(gt[img_index])

        # based on sample rate, decide whether this frame is sampled
        if img_index % best_sample_rate >= 1:
            # this frame is not sampled, so reuse the last saved
            # detection result
            dt_boxes_final = [box for box in save_dt]

        else:
            # this frame is sampled, so use the full model result
            dt_boxes_final = [box for box in current_full_model_dt]
            save_dt = [box for box in dt_boxes_final]

        for idx, box in enumerate(current_gt):
            current_gt[idx] = scale(current_gt[idx],
                                    original_config.resolution,
                                    best_config.resolution)

        tp[img_index], fp[img_index], fn[img_index] = \
            eval_single_image(current_gt, dt_boxes_final)

    tp_total = sum(tp.values())
    fp_total = sum(fp.values())
    fn_total = sum(fn.values())

    return compute_f1(tp_total, fp_total, fn_total), bw/origin_bw


def crop_image(img_in, xmin, ymin, xmax, ymax, w_out, h_out, img_out):
    cmd = ['ffmpeg',  '-y', '-hide_banner', '-i', img_in, '-vf',
           'crop={}:{}:{}:{},pad={}:{}:0:0:black'
           .format(xmax-xmin, ymax-ymin, xmin, ymin, w_out, h_out), img_out]
    subprocess.run(cmd)
