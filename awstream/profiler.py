""" the functions needed to profile and evaluate in Awstream """
import os
import subprocess
import copy
import pdb
from collections import defaultdict
from utils.model_utils import eval_single_image
from utils.utils import interpolation, compute_f1
from constants import RESOL_DICT


class VideoConfig:
    """ VideoConfig used in Awstream """
    def __init__(self, resolution, fps, quantizer=25):
        self.resolution = resolution
        self.fps = fps
        self.quantizer = quantizer

    def debug_print(self):
        """ print the detail of a config """
        print('resolution={}, fps={}, quantizer={}'
              .format(self.resolution, self.fps, self.quantizer))


def scale(box, in_resol, out_resol):
    """
    box: [x, y, w, h]
    in_resl: (width, height)
    out_resl: (width, height)
    """
    assert len(box) >= 4
    ret_box = box.copy()
    x_scale = out_resol[0]/in_resol[0]
    y_scale = out_resol[1]/in_resol[1]
    ret_box[0] = int(box[0] * x_scale)
    ret_box[1] = int(box[1] * y_scale)
    ret_box[2] = int(box[2] * x_scale)
    ret_box[3] = int(box[3] * y_scale)
    return ret_box


def scale_boxes(boxes, in_resol, out_resol):
    """ scale a list of boxes """
    return [scale(box, in_resol, out_resol) for box in boxes]


def compress_images_to_video(list_file: str, frame_rate: str, resolution: str,
                             quality: int, output_name: str):
    ''' Compress a set of frames to a video.  '''
    cmd = ['ffmpeg', '-y', '-loglevel', 'panic', '-r', str(frame_rate), '-f',
           'concat', '-safe', '0', '-i', list_file, '-s', str(resolution),
           '-vcodec', 'libx264', '-crf', str(quality), '-pix_fmt', 'yuv420p',
           '-hide_banner', output_name]
    subprocess.run(cmd, check=True)


def compute_video_size(video_name, img_path, frame_range,
                       original_config, target_config):
    """ return the size of the video compressed based on the parameters """
    image_resolution = target_config.resolution
    # Create a tmp list file contains all the selected iamges
    tmp_list_file = video_name + '_list.txt'
    sample_rate = original_config.fps/target_config.fps
    with open(tmp_list_file, 'w') as f_list:
        for img_index in range(frame_range[0], frame_range[1]+1):
            # based on sample rate, decide whether this frame is sampled
            if img_index % sample_rate >= 1:
                continue
            line = 'file \'{}/{:06}.jpg\'\n'.format(img_path, img_index)
            f_list.write(line)

    # compress the sampled image into a video
    frame_size = str(image_resolution[0]) + 'x' + str(image_resolution[1])
    output_video = video_name+'.mp4'
    compress_images_to_video(tmp_list_file, target_config.fps, frame_size,
                             target_config.quantizer, output_video)
    # get the video size
    video_size = os.path.getsize(output_video)
    # os.remove(output_video)
    os.remove(tmp_list_file)
    print('target fps={}, target resolution={}, video size={}'
          .format(target_config.fps, image_resolution, video_size))
    return video_size


def eval_images(image_range, gtruth, full_model_dt, original_config,
                target_config):
    """ evaluate the tp, fp, fn of a range of images """
    sample_rate = round(original_config.fps/target_config.fps)
    tpos = defaultdict(int)
    fpos = defaultdict(int)
    fneg = defaultdict(int)
    save_dt = []
    # pdb.set_trace()
    for idx in range(image_range[0], image_range[1]+1):
        if idx not in full_model_dt or idx not in gtruth:
            continue
        current_full_model_dt = copy.deepcopy(full_model_dt[idx])
        current_gt = copy.deepcopy(gtruth[idx])

        # based on sample rate, decide whether this frame is sampled
        if idx % sample_rate >= 1:
            # so reuse the last saved if no sampled
            dt_boxes_final = copy.deepcopy(save_dt)
        else:
            # sampled, so use the full model result
            dt_boxes_final = copy.deepcopy(current_full_model_dt)
            save_dt = copy.deepcopy(dt_boxes_final)

        # scale gt boxes
        current_gt = scale_boxes(current_gt, original_config.resolution,
                                 target_config.resolution)
        tpos[idx], fpos[idx], fneg[idx] = \
            eval_single_image(current_gt, dt_boxes_final)

        # print(idx, tpos[idx], fpos[idx], fneg[idx])
    return sum(tpos.values()), sum(fpos.values()), sum(fneg.values())


def find_target_fps(f1_list, fps_list, target_f1):
    """ use interpolation to find the ideal fps at target f1 """
    if f1_list[-1] < target_f1:
        target_fps = None
    else:
        index = next(x[0] for x in enumerate(f1_list)
                     if x[1] > target_f1)
        if index == 0:
            target_fps = fps_list[0]
        else:
            point_a = (f1_list[index-1], fps_list[index-1])
            point_b = (f1_list[index], fps_list[index])
            target_fps = interpolation(point_a, point_b, target_f1)
    return target_fps


def profile(video_name, dt_dict, original_config, frame_range,
            f_profile, resolution_list, temporal_sampling_list, target_f1=0.9):
    """ profile the combinations of fps and resolution
        return a list of config that satisfys the requirements """
    result = []
    gtruth = dt_dict[str(original_config.resolution[1]) + 'p']

    for resolution in resolution_list:
        # choose resolution
        f1_list = []

        if resolution not in dt_dict:
            continue
        print('profile [{}, {}], resolution={}, orginal resolution={}'
              .format(frame_range[0], frame_range[1], resolution,
                      original_config.resolution))

        for sample_rate in temporal_sampling_list:
            # choose frame rate
            target_config = VideoConfig(RESOL_DICT[resolution],
                                        original_config.fps/sample_rate)

            tp_total, fp_total, fn_total = eval_images(frame_range, gtruth,
                                                       dt_dict[resolution],
                                                       original_config,
                                                       target_config)
            f1_score = compute_f1(tp_total, fp_total, fn_total)

            f_profile.write(','.join([video_name, resolution, str(sample_rate),
                                      str(f1_score), str(tp_total),
                                      str(fp_total), str(fn_total)])+'\n')
            print('profile on {} {}, resolution={},sample rate={}, f1={}'
                  .format(video_name, frame_range, resolution, sample_rate,
                          f1_score))
            f1_list.append(f1_score)

        fps_list = [original_config.fps/x for x in temporal_sampling_list]

        target_fps = find_target_fps(f1_list, fps_list, target_f1)
        print("Resolution={} and target fps={}".format(resolution, target_fps))

        if target_fps is not None:
            tmp = copy.deepcopy(original_config)
            tmp.fps = target_fps
            tmp.resolution = RESOL_DICT[resolution]
            result.append(tmp)
    return result


def select_best_config(video, img_path_dict, original_config, configs,
                       frame_range):
    """ select the best config from a list of configs """
    resol = str(original_config.resolution[1]) + 'p'
    best_config = original_config
    original_bw = compute_video_size(video, img_path_dict[resol], frame_range,
                                     original_config, original_config)
    min_bw = original_bw

    for config in configs:
        target_fps = config.fps

        if target_fps is None:
            continue
        resol = str(config.resolution[1]) + 'p'
        bndwdth = compute_video_size(video, img_path_dict[resol], frame_range,
                                     original_config, config)

        if min_bw is None:
            min_bw = bndwdth

        elif bndwdth <= min_bw:
            best_config = config
            min_bw = bndwdth

    return best_config, min_bw/original_bw


def profile_eval(video_name, img_path_dict, gtruth, dts,
                 original_config, best_config, frame_range):
    """ evaluate the performance of best config """
    original_resol = str(original_config.resolution[1]) + 'p'
    best_resol = str(best_config.resolution[1]) + 'p'

    origin_bw = compute_video_size(video_name, img_path_dict[original_resol],
                                   frame_range, original_config,
                                   original_config)
    bndwdth = compute_video_size(video_name, img_path_dict[best_resol],
                                 frame_range, original_config, best_config)

    tp_total, fp_total, fn_total = eval_images(frame_range, gtruth, dts,
                                               original_config, best_config)

    return compute_f1(tp_total, fp_total, fn_total), bndwdth/origin_bw

# tpos = defaultdict(int)
# fpos = defaultdict(int)
# fneg = defaultdict(int)
# save_dt = []
# for img_index in range(start, end+1):
#     dt_boxes_final = []
#     if img_index not in full_model_dt or img_index not in gt:
#         continue
#     current_full_model_dt = copy.deepcopy(full_model_dt[img_index])
#     current_gt = copy.deepcopy(gt[img_index])
#
#     # based on sample rate, decide whether this frame is sampled
#     if img_index % sample_rate >= 1:
#         # so reuse the last saved if no sampled
#         dt_boxes_final = copy.deepcopy(save_dt)
#     else:
#         # sampled, so use the full model result
#         dt_boxes_final = copy.deepcopy(current_full_model_dt)
#         save_dt = copy.deepcopy(dt_boxes_final)
#
#     # scale gt boxes
#     for idx, box in enumerate(current_gt):
#         current_gt[idx] = scale(box, original_config.resolution,
#                                 RESOL_DICT[resolution])
#     tpos[img_index], fpos[img_index], fneg[img_index] = \
#         eval_single_image(current_gt, dt_boxes_final)
#
#     print(img_index, tpos[img_index], fpos[img_index],
#           fneg[img_index])
#     # pdb.set_trace()
# tp_total = sum(tpos.values())
# fp_total = sum(fpos.values())
# fn_total = sum(fneg.values())

# def crop_image(img_in, xmin, ymin, xmax, ymax, w_out, h_out, img_out):
#     cmd = ['ffmpeg',  '-y', '-hide_banner', '-i', img_in, '-vf',
#            'crop={}:{}:{}:{},pad={}:{}:0:0:black'
#            .format(xmax-xmin, ymax-ymin, xmin, ymin, w_out, h_out), img_out]
#     subprocess.run(cmd)


# tp = defaultdict(int)
# fp = defaultdict(int)
# fn = defaultdict(int)
# save_dt = []
# best_sample_rate = original_config.fps / best_config.fps
#
# for img_index in range(start, end+1):
#     if img_index not in gt or img_index not in dt:
#         continue
#     dt_boxes_final = []
#     current_full_model_dt = copy.deepcopy(dt[img_index])
#     current_gt = copy.deepcopy(gt[img_index])
#
#     # based on sample rate, decide whether this frame is sampled
#     if img_index % best_sample_rate >= 1:
#         # this frame is not sampled, so reuse the last saved
#         # detection result
#         dt_boxes_final = [box for box in save_dt]
#
#     else:
#         # this frame is sampled, so use the full model result
#         dt_boxes_final = [box for box in current_full_model_dt]
#         save_dt = [box for box in dt_boxes_final]
#
#     for idx, box in enumerate(current_gt):
#         current_gt[idx] = scale(current_gt[idx],
#                                 original_config.resolution,
#                                 best_config.resolution)
#
#     tp[img_index], fp[img_index], fn[img_index] = \
#         eval_single_image(current_gt, dt_boxes_final)
#
# tp_total = sum(tp.values())
# fp_total = sum(fp.values())
# fn_total = sum(fn.values())
