import csv
import copy
from collections import defaultdict
from benchmarking.constants import MODEL_COST
from benchmarking.utils.utils import compute_f1
from benchmarking.utils.model_utils import eval_single_image
class ModelSelection():
    """ModelSelection Pipeline."""

    def __init__(self, model_list, 
                 profile_log, target_f1=0.9):
        """Load the configs."""
        self.target_f1 = target_f1
        self.model_list = model_list
        self.profile_writer = csv.writer(open(profile_log, 'w', 1))
        self.profile_writer.writerow(
            ["video_name", "model", "f1", "tp", "fp", "fn"])

    def profile(self, video_name, video_dict, original_video, frame_range):
        """Profile on a set of model candidates.

        Return a list of config that satisfys the requirements.
        """
        f1_list = []

        for model in self.model_list:    
            video = video_dict[model]
            print('profile [{}, {}], model={}'
                  .format(frame_range[0], frame_range[1], model))

            tp_total, fp_total, fn_total = eval_images(frame_range, original_video, video)
            f1_score = compute_f1(tp_total, fp_total, fn_total)
            self.profile_writer.writerow([video_name, model,
                                              f1_score])
            print('profile on {} {}, model={},  f1={}'
                      .format(video_name, frame_range, model,
                              f1_score))
            f1_list.append(f1_score)


        # find the target model
        best_model = find_target_model(f1_list, self.model_list, self.target_f1)
        return best_model

    def evaluate(self, original_video, videos, best_model, frame_range):
        """Evaluate the performance of best config."""
        tp_total, fp_total, fn_total = eval_images(frame_range, original_video, videos[best_model])
        f1_score = compute_f1(tp_total, fp_total, fn_total)
        # f1 = compute_f1(frame_range, original_video, videos[best_model])
        gpu = MODEL_COST[best_model]/float(MODEL_COST['FasterRCNN'])
        return f1_score, gpu


# def compute_f1(frame_range, original_video, video):
#     gt = original_video.get_video_classification_label()
#     dt = video.get_video_classification_label()
#     tp_cn = 0
#     cn = 0
#     for frame_idx in range(frame_range[0], frame_range[1]):
#         cn += 1
#         if gt[frame_idx][0] == dt[frame_idx][0]:
#             tp_cn += 1
#     return tp_cn / float(cn)

    



def eval_images(image_range, original_video, video):
    """Evaluate the tp, fp, fn of a range of images."""
    # sample_rate = round(original_config.fps/target_config.fps)
    tpos = defaultdict(int)
    fpos = defaultdict(int)
    fneg = defaultdict(int)
    gtruth = original_video.get_video_detection()
    dets = video.get_video_detection()

    for idx in range(image_range[0], image_range[1] + 1):
        if idx not in dets or idx not in gtruth:
            continue
        current_dt = copy.deepcopy(dets[idx])
        current_gt = copy.deepcopy(gtruth[idx])

        tpos[idx], fpos[idx], fneg[idx] = \
            eval_single_image(current_dt, current_gt)

        # print(idx, tpos[idx], fpos[idx], fneg[idx])
    return sum(tpos.values()), sum(fpos.values()), sum(fneg.values())


def find_target_model(f1_list, model_list, target_f1):
    index = next(x[0] for x in enumerate(f1_list)
                         if x[1] >= target_f1)
    return model_list[index]
