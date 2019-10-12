""" Feature Scanning Scirpt """
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import load_metadata
# from ground_truth.video_feature_KITTI import compute_para
# from analyze_30s_feature import parse_feature_file, parse_feature_line
from feature_analysis.helpers import load_video_features, nonzero


# DATASET = 'validation_0000_segment-8888517708810165484_1549_770_1569_770_FRONT'
DATASET = 't_crossroad'
# DATASET = 'highway'
# DATASET = 'motorway'
# DATASET = 'driving1'
# DATASET = 'park'
# DATASET = 'driving_downtown'
# PATH = '/mnt/data/zhujun/dataset/Youtube/{}/2160p/profile/'.format(DATASET)
PATH = '/mnt/data/zhujun/dataset/Youtube/{}/1080p/profile/'.format(DATASET)
# PATH = '/mnt/data/zhujun/dataset/Youtube/{}/720p/profile/'.format(DATASET)
METADATA_FILE = '/mnt/data/zhujun/dataset/Youtube/{}/metadata.json'.format(
        DATASET)
SAMPLE_EVERY_N_FRAME_LIST = [1, 101]
# SHORT_VIDEO_LENGTH = 30  # seconds
# [10, 30, 100, 300, 1000]
# Every N frames sample once


def plot_cdf(data, num_bins, title, legend, xlabel, xlim=None):
    """ Use the histogram function to bin the data """
    counts, bin_edges = np.histogram(data, bins=num_bins, density=True)
    d_x = bin_edges[1] - bin_edges[0]
    # Now find the cdf
    cdf = np.cumsum(counts) * d_x
    # And finally plot the cdf
    plt.plot(bin_edges[1:], cdf, label=legend)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.ylim([0, 1.1])
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()


def main():
    parser = argparse.ArgumentParser(description="Feature scanning script")
    # parser.add_argument("--path", type=str, help="path contains all datasets")
    # parser.add_argument("--video", type=str, help="video name")
    parser.add_argument("--metadata", type=str, default='', required=True,
                        help="metadata file in Json")
    parser.add_argument("--feature_file", type=str, required=True,
                        help="output result file")
    # parser.add_argument("--output", type=str, help="output result file")
    # parser.add_argument("--log", type=str, help="log middle file")
    parser.add_argument("--short_video_length", type=int, required=True,
                        help="short video length in seconds")
    # parser.add_argument("--offset", type=int,
    #                     help="offset from beginning of the video in seconds")

    args = parser.parse_args()
    # path = args.path
    # dataset = args.video
    # output_file = args.output
    feature_file = args.feature_file
    short_video_length = args.short_video_length
    # offset = args.offset

    # metadata = load_metadata(path + video_name + '/metadata.json')
    metadata = load_metadata(args.metadata)
    fps = metadata['frame rate']
    # image_resolution = metadata['resolution']
    frame_count = metadata['frame count']


    # plot_cdf(velo_30s, 10000, '{}: object velocity'.format(DATASET),
    #          'original', 'object velocity', [1, 2.2])

    frame_features = load_video_features(feature_file)

    for n_frame in SAMPLE_EVERY_N_FRAME_LIST:
        short_vid_to_velo = defaultdict(list)
        short_vid_to_frame_id = defaultdict(list)
        sampled_velos = []
        for frame_idx in range(1, frame_count+1):
            short_vid = (frame_idx-1)//(fps * short_video_length)
            if frame_idx % n_frame == 0 and frame_idx in frame_features:
                # print(frame_features[frame_idx])
                sampled_velos.extend(frame_features[frame_idx]['Object Velocity'])
                short_vid_to_velo[short_vid].extend(frame_features[frame_idx]['Object Velocity'])
                short_vid_to_frame_id[short_vid].append(frame_idx)

        sampled_vals = list()
        sampled_medians = list()
        print(len(short_vid_to_velo.keys()))
        for key in sorted(short_vid_to_velo.keys()):
            sampled_vals.extend(nonzero(short_vid_to_velo[key]))
            # sampled_medians.append(np.median(nonzero(short_vid_to_velo[key])))
            sampled_medians.append(np.mean(nonzero(short_vid_to_velo[key])))
            # sampled_medians.append((np.quantile(nonzero(short_vid_to_velo[key]), 0.75)))
        # print(n_frame, len(short_vid_to_frame_id[key]))
        sampled_medians = [x for x in sampled_medians if not np.isnan(x)]
        print(len(sampled_vals))
        # print(sample_every_n_frame,
        #       len(short_vid_to_frame_id[key]),
        #       len(short_vid_to_velo[key])) #sampled_medians))

        plt.figure(1)
        plot_cdf(sampled_vals, 10000,
                 '{}: object velocity in 30s'.format(DATASET),
                 'every {} frames'.format(n_frame),
                 'object velocity', [1, 2.2])
        plt.figure(2)
        if n_frame == 1:
            plot_cdf(sampled_medians, 10000,
                     '{}: object velocity in 30s (median)'.format(DATASET),
                     'every {} frames'.format(n_frame),
                     'object velocity')
        else:
            plot_cdf(sampled_medians, 10000,
                     '{}: object velocity in 30s (median)'.format(DATASET),
                     'every {} frames'.format(100),
                     'object velocity')
            plt.axvline(x=1.2, c='k', linestyle='--')
            plt.axhline(y=0.34, c='k', linestyle='--')

    # plt.figure(1)
    # plot_cdf(nonzero(features[0]['object count']), 10000,
    #          '{}: object count'.format(DATASET), 'original', 'object count')

    # plt.figure(2)
    # plot_cdf(nonzero(features[0]['object area']), 10000,
    #          '{}: object area'.format(DATASET), 'original', 'object area')

    # plt.figure(3)
    # plot_cdf(nonzero(features[0]['arrival rate']), 10000,
    #          '{}: arrival rate'.format(DATASET), 'original', 'arrival rate',
    #          [0, 15])

    # plt.figure(4)
    # plot_cdf(nonzero(features[0]['total object area']), 10000,
    #          '{}: total object area'.format(DATASET),
    #          'original', 'total object area', [0, 0.2])

    # plt.figure(5)
    # plot_cdf(nonzero(features[0]['velocity']), 10000,
    #          '{}: object velocity'.format(DATASET),
    #          'original', 'object velocity', [0, 5])

    # plt.figure(6)
    # plot_cdf(cnt_30s, 10000, '{}: object count'.format(DATASET),
    #          'original', 'object count')
    # plt.figure(7)
    # plot_cdf(area_30s, 10000, '{}: object area'.format(DATASET),
    #          'original 30s median', 'object area')
    # plt.figure(8)
    # plot_cdf(arrival_rate_30s, 10000, '{}: arrival rate'.format(DATASET),
    #         'original', 'arrival rate')
    # plt.figure(9)
    # plot_cdf(tot_area_30s, 10000,
    #          '{}: total object area'.format(DATASET), 'original',
    #          'total object area')
    # plt.figure(10)
    # plot_cdf(velo_30s, 10000, '{}: object velocity'.format(DATASET),
    #          'original', 'object velocity', [1,2.2])
    # # sampled frames
    # for sample_every_n_frame in SAMPLE_EVERY_N_FRAME_LIST:
    #     sampled_frames = []
    #     for frame_idx in range(1, frame_count+1):
    #         if frame_idx % sample_every_n_frame == 0:
    #             # the frame is sampled
    #             sampled_frames.append(int(frame_idx - 0.1 * frame_rate))
    #             sampled_frames.append(frame_idx)

    #     # read annotations from ground truth file
    #     data = read_annot(annot_file, sampled_frames)
    #     paras = compute_para(data, image_resolution, frame_rate)

    #     all_filename = data[0]
    #     current_start = min(all_filename)
    #     current_end = max(all_filename)

    #     obj_cnt = []
    #     obj_area = []
    #     tot_area = []
    #     arrival_rate = []
    #     obj_velo = []
    #     short_video_id_to_cnt = defaultdict(list)
    #     short_video_id_to_area = defaultdict(list)
    #     short_video_id_to_arrival_rate = defaultdict(list)
    #     short_video_id_to_tot_area = defaultdict(list)
    #     short_video_id_to_velo = defaultdict(list)

    #     # Video feature distribution
    #     # for frame_id in range(1, frame_count + 1):
    #     #         # Find frames within a 30s short video
    #     #     if frame_id % sample_every_n_frame == 0 and frame_id in paras.num_of_objects:
    #     #         obj_cnt.append(paras.num_of_objects[frame_id])

    #     #     if frame_id % sample_every_n_frame == 0 and frame_id in paras.object_area:
    #     #         obj_area.extend([float(x) for x in
    #     #                          str(paras.object_area[frame_id]).split(' ')])
    #     #     if frame_id % sample_every_n_frame == 0 and frame_id in paras.arrival_rate:
    #     #         arrival_rate.append(paras.arrival_rate[frame_id])

    #     #     if frame_id % sample_every_n_frame == 0 and frame_id in paras.velocity and paras.velocity[frame_id]:
    #     #         obj_velo.extend([float(x) for x in
    #     #                          str(paras.velocity[frame_id]).split(' ')])

    #     #     if frame_id % sample_every_n_frame == 0 and frame_id in paras.total_object_area:
    #     #         tot_area.append(paras.total_object_area[frame_id])

    #     # 30s features distribution
    #     # Consecutive 30s windows
    #     # for frame_id in range(1, frame_count + 1):
    #     #     short_video_id = (frame_id-1)//(frame_rate * SHORT_VIDEO_LENGTH)
    #     #         # Find frames within a 30s short video
    #     #     if frame_id % sample_every_n_frame == 0 and frame_id in paras.num_of_objects:
    #     #         short_video_id_to_cnt[short_video_id].append(paras.num_of_objects[frame_id])

    #     #     if frame_id % sample_every_n_frame == 0 and frame_id in paras.object_area:
    #     #         short_video_id_to_area[short_video_id].extend([float(x) for x in
    #     #                          str(paras.object_area[frame_id]).split(' ')])
    #     #     if frame_id % sample_every_n_frame == 0 and frame_id in paras.arrival_rate:
    #     #         short_video_id_to_arrival_rate[short_video_id].append(paras.arrival_rate[frame_id])

    #     #     if frame_id % sample_every_n_frame == 0 and frame_id in paras.velocity and paras.velocity[frame_id]:
    #     #         short_video_id_to_velo[short_video_id].extend([float(x) for x in
    #     #                          str(paras.velocity[frame_id]).split(' ')])

    #     #     if frame_id % sample_every_n_frame == 0 and frame_id in paras.total_object_area:
    #     #         short_video_id_to_tot_area[short_video_id].append(paras.total_object_area[frame_id])

    #     # 30s features distribution
    #     # 30s sliding windows
    #     short_video_id = 0
    #     for frame_id in range(1, frame_count + 1 - frame_rate * SHORT_VIDEO_LENGTH):
    #         # Find frames within a 30s short video
    #         for i in range(frame_id, frame_id + frame_rate * SHORT_VIDEO_LENGTH):
    #             if i % sample_every_n_frame == 0 and i in paras.num_of_objects:
    #                 short_video_id_to_cnt[short_video_id].append(paras.num_of_objects[i])

    #             if i % sample_every_n_frame == 0 and i in paras.object_area:
    #                 short_video_id_to_area[short_video_id].extend([float(x) for x in
    #                                  str(paras.object_area[i]).split(' ')])
    #             if i % sample_every_n_frame == 0 and i in paras.arrival_rate:
    #                 short_video_id_to_arrival_rate[short_video_id].append(paras.arrival_rate[i])

    #             if i % sample_every_n_frame == 0 and i in paras.velocity and paras.velocity[i]:
    #                 short_video_id_to_velo[short_video_id].extend([float(x) for x in
    #                                  str(paras.velocity[i]).split(' ')])

    #             if i % sample_every_n_frame == 0 and i in paras.total_object_area:
    #                 short_video_id_to_tot_area[short_video_id].append(paras.total_object_area[i])
    #         short_video_id = short_video_id + 1
    #     # import pdb
    #     # pdb.set_trace()

    #     plt.figure(1)
    #     plot_cdf(nonzero(obj_cnt), 10000, '{}: object count'.format(DATASET),
    #              'every {} frames'.format(sample_every_n_frame),
    #              'object count')

    #     plt.figure(2)
    #     plot_cdf(nonzero(obj_area), 10000, '{}: object area'.format(DATASET),
    #              'every {} frames'.format(sample_every_n_frame),
    #              'object area')
    #     plt.figure(3)
    #     plot_cdf(nonzero(arrival_rate), 10000,
    #              '{}: arrival rate'.format(DATASET),
    #              'every {} frames'.format(sample_every_n_frame),
    #              'arrival rate', [0, 15])
    #     plt.figure(4)
    #     plot_cdf(nonzero(tot_area), 10000,
    #              '{}: total object area'.format(DATASET),
    #              'every {} frames'.format(sample_every_n_frame),
    #              'total object area', [0, 0.2])
    #     plt.figure(5)
    #     plot_cdf(nonzero(obj_velo), 10000,
    #              '{}: object velocity'.format(DATASET),
    #              'every {} frames'.format(sample_every_n_frame),
    #              'object velocity', [0, 5])

    #     plt.figure(6)
    #     sampled_medians = list()
    #     for key in sorted(short_video_id_to_cnt.keys()):
    #         #  print(len(short_video_id_to_cnt[key]))
    #         sampled_medians.append((np.median(nonzero(short_video_id_to_cnt[key]))))
    #         # sampled_medians.append((np.mean(nonzero(short_video_id_to_cnt[key]))))
    #         # sampled_medians.append((np.quantile(nonzero(short_video_id_to_cnt[key]), 0.75)))
    #     sampled_medians = [x for x in sampled_medians if not np.isnan(x)]
    #     plot_cdf(sampled_medians, 10000,
    #              '{}: object count in 30s median'.format(DATASET),
    #              'every {} frames'.format(sample_every_n_frame),
    #              'object count')

    #     plt.figure(7)
    #     sampled_medians = list()
    #     for key in sorted(short_video_id_to_area.keys()):
    #         #  print(len(short_video_id_to_area[key]))
    #         sampled_medians.append((np.median(nonzero(short_video_id_to_area[key]))))
    #         # sampled_medians.append((np.mean(nonzero(short_video_id_to_area[key]))))
    #         # sampled_medians.append((np.quantile(nonzero(short_video_id_to_area[key]), 0.75)))
    #     sampled_medians = [x for x in sampled_medians if not np.isnan(x)]
    #     plot_cdf(sampled_medians, 10000, '{}: object area in 30s (median)'.format(DATASET),
    #              'every {} frames'.format(sample_every_n_frame),
    #              'object area')

    #     plt.figure(8)
    #     sampled_medians = list()
    #     for key in sorted(short_video_id_to_arrival_rate.keys()):
    #         sampled_medians.append((np.median(nonzero(short_video_id_to_arrival_rate[key]))))
    #         # sampled_medians.append((np.mean(nonzero(short_video_id_to_arrival_rate[key]))))
    #         # sampled_medians.append((np.quantile(nonzero(short_video_id_to_arrival_rate[key]), 0.75)))
    #     sampled_medians = [x for x in sampled_medians if not np.isnan(x)]
    #     plot_cdf(sampled_medians, 10000, '{}: arrival rate in 30s (median)'.format(DATASET),
    #              'every {} frames'.format(sample_every_n_frame),
    #              'arrival rate')

    #     plt.figure(9)
    #     sampled_medians = list()
    #     for key in sorted(short_video_id_to_tot_area.keys()):
    #         sampled_medians.append((np.median(nonzero(short_video_id_to_tot_area[key]))))
    #         # sampled_medians.append((np.mean(nonzero(short_video_id_to_tot_area[key]))))
    #         # sampled_medians.append((np.quantile(nonzero(short_video_id_to_tot_area[key]), 0.75)))
    #     sampled_medians = [x for x in sampled_medians if not np.isnan(x)]
    #     plot_cdf(sampled_medians, 10000,
    #              '{}: total object area in 30s (median)'.format(DATASET),
    #              'every {} frames'.format(sample_every_n_frame),
    #              'total object area in 30s')
    #     plt.figure(10)
    #     sampled_medians = list()
    #     for key in sorted(short_video_id_to_velo.keys()):
    #         sampled_medians.append((np.median(nonzero(short_video_id_to_velo[key]))))
    #         # sampled_medians.append((np.mean(nonzero(short_video_id_to_velo[key]))))
    #         # sampled_medians.append((np.quantile(nonzero(short_video_id_to_velo[key]), 0.75)))
    #     sampled_medians = [x for x in sampled_medians if not np.isnan(x)]
    #     # import pdb
    #     # pdb.set_trace()
    #     plot_cdf(sampled_medians, 10000,
    #              '{}: object velocity in 30s (median)'.format(DATASET),
    #              'every {} frames'.format(sample_every_n_frame),
    #              'object velocity')
    #     # plt.axvline(x=1.2, color='k', linestyle='--')
    #     # plt.axhline(y=0.58, color='k', linestyle='--')
    #     #  plt.axvline(x=1.7, color='k', linestyle='--')
    #     #  plt.axhline(y=0.991, color='k', linestyle='--')

    # #  plt.figure(1)
    # #  plt.savefig('figs/obj_cnt_cdf.png')
    # #  plt.figure(2)
    # #  plt.savefig('figs/obj_area_cdf.png')
    # #  plt.figure(3)
    # #  plt.savefig('figs/arrival_rate_cdf.png')
    # #  plt.figure(4)
    # #  plt.savefig('figs/tot_area_cdf.png')
    # #  plt.figure(5)
    # #  plt.savefig('figs/obj_velo_cdf.png')
    # #  plt.figure(6)
    # #  plt.savefig('figs/obj_cnt_median_cdf.png')
    # #  plt.figure(7)
    # #  plt.savefig('figs/obj_area_median_cdf.png')
    # #  plt.figure(8)
    # #  plt.savefig('figs/arrival_rate_median_cdf.png')
    # #  plt.figure(9)
    # #  plt.savefig('figs/tot_area_median_cdf.png')
    # #  plt.figure(10)
    # #  plt.savefig('figs/obj_velo_median_cdf.png')

    plt.show()

    #  print(short_video_id_to_velo)


if __name__ == '__main__':
    main()
