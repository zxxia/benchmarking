import os
import matplotlib.pyplot as plt
from constants import CAMERA_TYPES, COCOLabels, RESOL_DICT
from utils.utils import load_metadata
from utils.model_utils import load_full_model_detection, \
    filter_video_detections, remove_overlappings
from feature_analysis.helpers import sample_frames, get_areas, plot_cdf

ROOT = '/data/zxxia/benchmarking/results/videos'


def load_detections(video, dt_file, resol):
    """ load and filter detections """
    dts, nb_frame = load_full_model_detection(dt_file)
    if video in CAMERA_TYPES['moving']:
        dts = filter_video_detections(dts,
                                      target_types={COCOLabels.CAR.value,
                                                    COCOLabels.BUS.value,
                                                    # COCOLabels.TRAIN.value,
                                                    COCOLabels.TRUCK.value
                                                    },
                                      height_range=(RESOL_DICT[resol][1]//20,
                                                    RESOL_DICT[resol][1]),
                                      )  # score_range=(0.5, 1)
    else:
        dts = filter_video_detections(dts,
                                      target_types={COCOLabels.CAR.value,
                                                    COCOLabels.BUS.value,
                                                    # COCOLabels.TRAIN.value,
                                                    COCOLabels.TRUCK.value
                                                    },
                                      width_range=(0, RESOL_DICT[resol][0]/2),
                                      height_range=(RESOL_DICT[resol][0]//20,
                                                    RESOL_DICT[resol][0]/2))
    if video == 'road_trip':
        for frame_idx in dts:
            tmp_boxes = []
            for box in dts[frame_idx]:
                xmin, ymin, xmax, ymax = box[:4]
                if ymin >= 500/720*RESOL_DICT[resol][1] or \
                        ymax >= 645 / 720 * RESOL_DICT[resol][1]:
                    continue
                if (xmax - xmin) >= 2/3 * RESOL_DICT[resol][0]:
                    continue
                tmp_boxes.append(box)
            dts[frame_idx] = tmp_boxes
    for frame_idx, bboxes in dts.items():
        # merge all vehicle labels into CAR
        for box_pos, box in enumerate(bboxes):
            box[4] = COCOLabels.CAR.value
            bboxes[box_pos] = box
        dts[frame_idx] = bboxes
        # remove overlappings to mitigate occultation
        dts[frame_idx] = remove_overlappings(bboxes, 0.3)

    return dts, nb_frame


def main():
    video = 'road_trip'
    video = 'driving2'
    video = 'driving_downtown'
    resol = '720p'

    metadata_file = '/data/zxxia/videos/{}/metadata.json'.format(video)
    metadata = load_metadata(metadata_file)
    dt_file = os.path.join(ROOT, video, resol, 'profile',
                           'updated_gt_FasterRCNN_COCO_no_filter.csv')
    dts, _ = load_detections(video, dt_file, '720p')

    dt_file = os.path.join(ROOT, video, resol, 'profile',
                           'updated_gt_mobilenet_COCO_no_filter.csv')
    dts_mobilenet, _ = load_detections(video, dt_file, resol)
    nframe = 50
    dts_mobilenet_sampled = sample_frames(dts_mobilenet,
                                          metadata['frame count'], nframe)

    areas = get_areas(dts, 1, metadata['frame count'], resol)
    areas_sampled = get_areas(dts, 1, metadata['frame count'], resol)
    areas_mobilenet = get_areas(
        dts_mobilenet, 1, metadata['frame count'], resol)
    areas_mobilenet_sampled = get_areas(dts_mobilenet_sampled, 1,
                                        metadata['frame count'], resol)
    # print(len(dts_mobilenet), len(dts_mobilenet_sampled))
    plot_cdf(areas, 1000, 'frcnn')
    # plot_cdf(areas_sampled, 1000, 'frcnn sampled {}'.format(nframe))
    plot_cdf(areas_mobilenet, 1000, 'mobilenet')
    plot_cdf(areas_mobilenet_sampled, 1000,
             'mobilenet sampled {}'.format(nframe))

    nframe = 100
    dts_mobilenet_sampled = sample_frames(dts_mobilenet,
                                          metadata['frame count'], nframe)
    areas_mobilenet_sampled = get_areas(dts_mobilenet_sampled, 1,
                                        metadata['frame count'], resol)
    # plot_cdf(areas_mobilenet_sampled, 1000,
    #          'mobilenet sampled {}'.format(nframe))
    plt.xlabel('object size')
    plt.ylabel('CDF')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
