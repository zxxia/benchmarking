"""Generate video analytic pipeline relative performance figure."""
import csv
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from benchmarking.awstream.Awstream import load_awstream_results
# from benchmarking.chameleon.Chameleon import load_chameleon_results
# from benchmarking.glimpse.Glimpse import load_glimpse_results
# from benchmarking.noscope.NoScope import load_noscope_results
from benchmarking.videostorm.VideoStorm import load_videostorm_e2e_results
# from benchmarking.vigil.Vigil import load_vigil_results

VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4', 'drift',
          'driving1', 'driving_downtown', 'highway',
          'nyc',  'lane_split',  # 'driving2','jp',
          'motorway', 'park',  # 'russia', 'russia1', 'traffic',  'tw',   'tw1',
          'tw_under_bridge']

PS = 20


def videostorm_vs_glimpse():
    # gl_path = '/home/zxxia/Projects/benchmarking/glimpse/'\
    #     'glimpse_e2e_frame_select_results/glimpse_frame_select_{}.csv'
    # gl_path = '/home/zxxia/Projects/benchmarking/glimpse/'\
    #     'glimpse_e2e_perfect_tracking_results/glimpse_perfect_tracking_{}.csv'
    # gl_path = '/home/zxxia/Projects/benchmarking/glimpse/'\
    #     'glimpse_e2e_tracking_results/glimpse_tracking_{}.csv'
    # gl_path = '/data/zxxia/benchmarking/glimpse/'\
    #     'e2e_results/glimpse_tracking_{}.csv'
    vs_path = '/data/zxxia/benchmarking/results/videostorm/'\
        'test_coverage_results/videostorm_coverage_results_{}.csv'
    # plt.figure()
    # videos_to_plot = []
    # vs_perfs_to_plot = []
    # vs_accs_to_plot = []
    # for video in VIDEOS:
    #     filename = vs_path.format(video)
    #     videos, models, perfs, frame_rates, accs = load_videostorm_e2e_results(
    #         filename)
    #     videos_to_plot.extend(videos)
    #     vs_perfs_to_plot.extend(perfs)
    #     vs_accs_to_plot.extend(accs)
    #     # print(videos)
    # print(len(videos_to_plot))
    # # plt.scatter(perfs_to_plot, accs_to_plot, s=PS, label='videostorm')
    ax = plt.gca()
    # videos_to_plot = []
    # gl_perfs_to_plot = []
    # gl_accs_to_plot = []
    # for video in VIDEOS:
    #     filename = gl_path.format(video)
    #     videos, perfs, accs = load_glimpse_results(filename)
    #     gl_perfs_to_plot.extend(perfs)
    #     gl_accs_to_plot.extend(accs)
    #     videos_to_plot.extend(videos)
    #     # print(videos)
    # perf_ratios = np.array(vs_perfs_to_plot) / np.array(gl_perfs_to_plot)
    # acc_ratios = np.array(vs_accs_to_plot)/np.array(gl_accs_to_plot)
    # print(len(videos_to_plot))
    # with open('gl_vs.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for perf_ratio, acc_ratio in zip(perf_ratios, acc_ratios):
    #         if perf_ratio > 0 and acc_ratio > 0:
    #             writer.writerow([perf_ratio, acc_ratio])
    gl_path = '/data/zxxia/benchmarking/glimpse/e2e_results_kcf/glimpse_result_{}.csv'
    perf_ratios = []
    acc_ratios = []
    videos_to_plot = []
    for video in VIDEOS:
        gl_results = pd.read_csv(gl_path.format(video))
        vs_results = pd.read_csv(vs_path.format(video))
        assert len(gl_results) == len(vs_results)
        perf_ratios += list(vs_results['gpu time'][gl_results.index].to_numpy() -
                            gl_results['frame rate'].to_numpy())
        acc_ratios += list(vs_results['f1'][gl_results.index].to_numpy() -
                           gl_results['f1'].to_numpy())
        videos_to_plot += gl_results['video chunk'].to_list()
    for perf, acc, vid in zip(perf_ratios, acc_ratios, videos_to_plot):
        if perf > 1.0 and acc < 1.0:
            ax.annotate(vid, (perf, acc))
        if perf < 1.0 and acc > 1.0:
            ax.annotate(vid, (perf, acc))
    plt.title('Glimpse vs. VideoStorm')
    plt.scatter(perf_ratios, acc_ratios, s=PS)
    plt.axvline(x=0, linestyle='--', c='k')
    plt.axhline(y=0, linestyle='--', c='k')
    plt.xlabel('GPU Processing time')
    plt.ylabel('Accuracy')
    with open('gl_vs_improved.csv', 'w') as f:
        writer = csv.writer(f)
        for perf_ratio, acc_ratio in zip(perf_ratios, acc_ratios):
            # if perf_ratio > 0 and acc_ratio > 0:
            writer.writerow([perf_ratio, acc_ratio])
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    # plt.ylim(0, 2)


# plt.figure()
# ax = plt.gca()
# aw_path = 'e2e_results_30s_10s/awstream_e2e_results_{}.csv'
# videos_to_plot = []
# awstream_bw_to_plot = []
# awstream_accs_to_plot = []
# for video in VIDEOS:
#     filename = aw_path.format(video)
#     videos, bw, accs = load_awstream_results(filename)
#     awstream_bw_to_plot.extend(bw)
#     awstream_accs_to_plot.extend(accs)
#     videos_to_plot.extend(videos)
#     print(video, len(videos))
#     # print(videos)
#
# print(len(videos_to_plot))
# plt.title('AWStream vs. Vigil')
# # plt.scatter(bw_to_plot, accs_to_plot, s=PS, label='AWStream')
# plt.xlabel('Network Bandwidth')
# plt.ylabel('Accuracy')
# plt.xlim(0, 2)
# plt.ylim(0, 2)
# plt.axvline(x=1, linestyle='--', c='k')
# plt.axhline(y=1, linestyle='--', c='k')
#
# # vigil_path = '/data/zxxia/benchmarking/Vigil/e2e_test_results/vigil_{}.csv'
# vigil_path = '/home/zxxia/Projects/benchmarking/vigil/vigil_e2e_results/vigil_{}.csv'
# videos_to_plot = []
# vigil_bw_to_plot = []
# vigil_accs_to_plot = []
# for video in VIDEOS:
#     filename = vigil_path.format(video)
#     videos, bw, accs = load_vigil_results(filename)
#     print(video, len(videos))
#     vigil_bw_to_plot.extend(bw)
#     vigil_accs_to_plot.extend(accs)
#     videos_to_plot.extend(videos)
#     # print(videos)
# print(len(videos_to_plot))
# bw_ratios = np.array(vigil_bw_to_plot) / np.array(awstream_bw_to_plot)
# acc_ratios = np.array(vigil_accs_to_plot) / np.array(awstream_accs_to_plot)
# plt.scatter(bw_ratios, acc_ratios, s=PS)
# plt.legend()
# with open('aw_vg.csv', 'w') as f:
#     writer = csv.writer(f)
#     for perf_ratio, acc_ratio in zip(bw_ratios, acc_ratios):
#         if perf_ratio > 0 and acc_ratio > 0:
#             writer.writerow([perf_ratio, acc_ratio])


# Plot noscope results
# plt.figure()
# VIDEOS = ['cropped_crossroad4',  # 'cropped_crossroad4_2',
#           'cropped_crossroad5', 'cropped_driving2']
# ax = plt.gca()
# aw_path = '/home/zxxia/Projects/benchmarking/noscope/noscope_result_{}.csv'
# videos_to_plot = []
# no_perf_to_plot = []
# no_accs_to_plot = []
# for video in VIDEOS:
#     filename = aw_path.format(video)
#     videos, perf, accs = load_noscope_results(filename)
#     no_perf_to_plot.extend(perf)
#     no_accs_to_plot.extend(accs)
#     videos_to_plot.extend(videos)
#     # print(videos)
#
# ax = plt.gca()
# ch_path = '/home/zxxia/Projects/benchmarking/chameleon/chameleon_e2e_result_{}.csv'
# videos_to_plot = []
# ch_perf_to_plot = []
# ch_accs_to_plot = []
# for video in VIDEOS:
#     filename = ch_path.format(video)
#     videos, models, perf, accs = load_chameleon_results(filename)
#     ch_perf_to_plot.extend(perf)
#     ch_accs_to_plot.extend(accs)
#     videos_to_plot.extend(videos)
#
# perf_ratios = np.array(ch_perf_to_plot) / np.array(no_perf_to_plot)
# acc_ratios = np.array(ch_accs_to_plot) / np.array(no_accs_to_plot)
# plt.title('Noscope vs. Chameleon')
# plt.scatter(perf_ratios, acc_ratios, s=PS)
# plt.xlabel('Compute Cost(GPU)')
# plt.ylabel('Accuracy')
# plt.axvline(x=1, linestyle='--', c='k')
# plt.axhline(y=1, linestyle='--', c='k')
# with open('no_ch.csv', 'w') as f:
#     writer = csv.writer(f)
#     for perf_ratio, acc_ratio in zip(perf_ratios, acc_ratios):
#         if perf_ratio > 0 and acc_ratio > 0:
#             writer.writerow([perf_ratio, acc_ratio])


def awstream_vs_glimpse():
    # Plot glimpse vs. awstream results
    plt.figure(0)
    ax = plt.gca()
    gl_path = '/data/zxxia/benchmarking/glimpse/e2e_results_kcf/glimpse_result_{}.csv'
    aws_path = '/data/zxxia/benchmarking/results/awstream/e2e_results_iframe_control/awstream_e2e_results_{}.csv'
    aws_bw = pd.read_csv(
        '/data/zxxia/benchmarking/results/awstream/e2e_results_iframe_control/eval_video_sizes.csv')
    bw_ratios = []
    acc_ratios = []
    videos_to_plot = []

    gl_bw_t = []
    aws_bw_t = []
    gl_f1_t = []
    aws_f1_t = []
    for video in ['crossroad3']:
        # for video in VIDEOS:
        gl_results = pd.read_csv(gl_path.format(video))
        aws_results = pd.read_csv(aws_path.format(video))

        # print(aws_bw[aws_bw['video'].str.contains('|'.join(gl_results['video chunk'].tolist())
        #                                           )].sort_values(by=['video']))
        # print(gl_results['video chunk'])
        for index, row in gl_results.iterrows():
            # print(row['video chunk'])
            target_index = aws_bw['video'].str.contains(
                row['video chunk']+'.mp4')
            # print(aws_bw[target_index]['video'])
            video_bw = aws_bw[target_index]['size'].iloc[0]
            target_index = aws_results['dataset'].str.contains(
                row['video chunk'])
            original_bw = video_bw / \
                aws_results[target_index]['bandwidth'].iloc[0]
            acc_ratios.append(
                row['f1']-aws_results[target_index]['f1'].iloc[0])
            bw_ratios.append(row['bw']/original_bw -
                             aws_results[target_index]['bandwidth'].iloc[0])
            videos_to_plot.append(row['video chunk'])
            if video == 'crossroad3':
                gl_bw_t.append(row['bw']/original_bw)
                aws_bw_t.append(aws_results[target_index]['bandwidth'].iloc[0])
                gl_f1_t.append(row['f1'])
                aws_f1_t.append(aws_results[target_index]['f1'].iloc[0])

    plt.axvline(x=0, linestyle='--', c='k')
    plt.axhline(y=0, linestyle='--', c='k')
    plt.scatter(bw_ratios, acc_ratios, s=PS)
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.15, 0.15)
    # plt.ylim(-0.5, 0.5)
    # for perf, acc, vid in zip(bw_ratios, acc_ratios, videos_to_plot):
    #     if (acc < 0 and perf > 0 or acc > 0 and perf < 0) and 'crossroad3' in vid:
    #         ax.annotate(vid, (perf, acc))
    ts = np.arange(len(gl_bw_t[31:35])) * 30
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(ts, gl_f1_t[31:35], 'o-', label='glimpse')
    plt.plot(ts, aws_f1_t[31:35], 'o-', label='awstream')
    plt.title('f1')
    plt.subplot(2, 1, 2)
    # gl_results = pd.read_csv(gl_path.format('crossroad3'))
    # aws_results = pd.read_csv(aws_path.format('crossroad3'))
    plt.plot(ts, gl_bw_t[31:35], 'o-', label='glimpse')
    plt.plot(ts, aws_bw_t[31:35], 'o-', label='awstream')
    plt.title('bandwidth')
    plt.legend()

    # with open('var_over_time.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for t, aws_acc, aws_perf, gl_acc, gl_perf in zip(
    #             ts, aws_f1_t[31:35], aws_bw_t[31:35], gl_f1_t[31:35], gl_bw_t[31:35]):
    #         writer.writerow([t, aws_acc, aws_perf, gl_acc, gl_perf])
    # plt.xlim(0, 2)
    # plt.ylim(0, 2)
    # with open('aw_gl.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for perf_ratio, acc_ratio in zip(bw_ratios, acc_ratios):
    #         # if perf_ratio > 0 and acc_ratio > 0:
    #         writer.writerow([perf_ratio, acc_ratio])


def main():
    # videostorm_vs_glimpse()
    awstream_vs_glimpse()
    plt.show()


if __name__ == '__main__':
    main()
