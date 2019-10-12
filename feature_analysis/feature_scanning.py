""" Feature Scanning Scirpt """
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import load_metadata
from feature_analysis.helpers import load_video_features, nonzero


DATASET = 't_crossroad'
PATH = '/mnt/data/zhujun/dataset/Youtube/{}/1080p/profile/'.format(DATASET)
# PATH = '/mnt/data/zhujun/dataset/Youtube/{}/720p/profile/'.format(DATASET)
METADATA_FILE = '/mnt/data/zhujun/dataset/Youtube/{}/metadata.json'.format(
        DATASET)
SAMPLE_EVERY_N_FRAME_LIST = [1, 101]


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


def sample_video_features(video_features, metadata, short_video_length,
                          sample_rate):
    """ Sample video features """
    fps = metadata['frame rate']
    # resolution = metadata['resolution']
    frame_cnt = metadata['frame count']
    short_vid_features = defaultdict(lambda: {'Object Count': [],
                                              'Object Area': [],
                                              'Total Area': [],
                                              'Object Velocity': []})
    short_vid_to_frame_id = defaultdict(list)
    # sampled_velos = []
    for fid in range(1, frame_cnt+1):
        short_vid = (fid-1)//(fps * short_video_length)
        if fid % sample_rate == 0 and fid in video_features:
            # print(frame_features[frame_idx])
            # sampled_velos.extend(video_features[fid]['Object Velocity'])
            short_vid_features[short_vid]['Object Velcity']\
                .extend(video_features[fid]['Object Velocity'])
            short_vid_features[short_vid]['Object Area'] \
                .extend(video_features[fid]['Object Area'])
            short_vid_features[short_vid]['Object Count'] \
                .append(video_features[fid]['Object Count'])
            short_vid_features[short_vid]['Total Area'] \
                .append(video_features[fid]['Total Area'])
            short_vid_to_frame_id[short_vid].append(fid)
    return short_vid_features, short_vid_to_frame_id


def main():
    """ feature scanning """
    parser = argparse.ArgumentParser(description="Feature scanning script")
    # parser.add_argument("--video", type=str, help="video name")
    parser.add_argument("--metadata", type=str, default='', required=True,
                        help="metadata file in Json")
    parser.add_argument("--feature_file", type=str, required=True,
                        help="feature file")
    # parser.add_argument("--output", type=str, help="output result file")
    # parser.add_argument("--log", type=str, help="log middle file")
    parser.add_argument("--short_video_length", type=int, required=True,
                        help="short video length in seconds")

    args = parser.parse_args()
    # dataset = args.video
    # output_file = args.output
    short_video_length = args.short_video_length

    metadata = load_metadata(args.metadata)

    # plot_cdf(velo_30s, 10000, '{}: object velocity'.format(DATASET),
    #          'original', 'object velocity', [1, 2.2])

    video_features = load_video_features(args.feature_file)

    for n_frame in SAMPLE_EVERY_N_FRAME_LIST:
        shvid_features, shvid_to_frame_id = \
            sample_video_features(video_features, metadata,
                                  short_video_length, n_frame)
        sampled_vals = list()
        sampled_medians = list()
        sampled_means = list()
        sampled_75_percentile = list()
        sampled_25_percentile = list()
        sampled_medians = list()
        print(len(shvid_features.keys()))
        for key in sorted(shvid_features.keys()):
            sampled_vals.extend(nonzero(shvid_features[key]['Object Velocity']))
            sampled_means.append(np.median(nonzero(shvid_features[key]['Object Velocity'])))
            sampled_medians.append(np.mean(nonzero(shvid_features[key]['Object Velocity'])))
            sampled_75_percentile.append(np.quantile(nonzero(shvid_features[key]['Object Velocity']), 0.75))
            sampled_25_percentile.append(np.quantile(nonzero(shvid_features[key]['Object Velocity']), 0.25))
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
    plt.show()


if __name__ == '__main__':
    main()
