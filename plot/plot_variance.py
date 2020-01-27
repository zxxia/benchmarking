"""Generate plots for VAP variance across videos and cost CDF"""
import csv
import pdb

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import pandas as pd

from benchmarking.awstream.Awstream import load_awstream_results
from benchmarking.glimpse.Glimpse import load_glimpse_results
# from benchmarking.noscope.Noscope import load_noscope_results
from benchmarking.videostorm.VideoStorm import load_videostorm_e2e_results


def eigsorted(cov):
    """Return sorted eigen values and vectors."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def get_cov_ellipse(cov, centre, nstd):
    """Return a matplotlib Ellipse patch.

    cov: covariance matrix
    cov centred at centre and scaled by the factor nstd.
    """
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(xy=centre, width=w, height=h, angle=theta)
    print('width', w, 'height', h, 'angle', theta)
    return ell


def plot_cdf(data, num_bins, legend):
    """Use the histogram function to bin the data."""
    counts, bin_edges = np.histogram(data, bins=num_bins, density=True)
    d_x = bin_edges[1] - bin_edges[0]
    # Now find the cdf
    cdf = np.cumsum(counts) * d_x
    # And finally plot the cdf
    plt.plot(bin_edges[1:], cdf, label=legend)
    return bin_edges[1:], cdf
    # plt.title(title)
    # plt.xlabel(xlabel)
    # plt.ylabel('CDF')
    # plt.ylim([0, 1.1])
    # plt.legend()


VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4', 'drift',
          'driving1', 'driving_downtown', 'highway',
          'nyc', 'jp',  'lane_split',  # 'driving2',
          'motorway', 'park', 'russia', 'russia1', 'traffic', 'tw', 'tw1',
          'tw_under_bridge']  #
NSTD = 2
PS = 2


def plot_videostorm():

    videos_to_plot = []
    perfs_to_plot = []
    accs_to_plot = []
    # vs_path = '/data/zxxia/benchmarking/videostorm/test_coverage_results/' \
    #     'videostorm_coverage_results_{}.csv'
    vs_path = '/home/zxxia/Projects/benchmarking/videostorm/test_coverage_results/' \
        'videostorm_coverage_results_{}.csv'
    for video in VIDEOS:
        filename = vs_path.format(video)
        videos, models, perfs, frame_rates, accs = load_videostorm_e2e_results(
            filename)
        videos_to_plot.extend(videos)
        perfs_to_plot.extend(perfs)
        accs_to_plot.extend(accs)
        # print(videos)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.scatter(perfs_to_plot, accs_to_plot, s=PS)
    ax = plt.gca()
    # for perf, acc, vid in zip(perfs_to_plot, accs_to_plot, videos_to_plot):
    #     ax.annotate(vid, (perf, acc))
    plt.xlabel('GPU Processing time')
    plt.ylabel('Accuracy')
    plt.title('VideoStorm')
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    avg_perf = np.mean(perfs_to_plot)
    std_perf = np.std(perfs_to_plot)
    med_perf = np.median(perfs_to_plot)

    avg_acc = np.mean(accs_to_plot)
    std_acc = np.std(accs_to_plot)
    med_acc = np.median(accs_to_plot)
    print('vs avg perf={:.4f}, avg acc={:.4f}, med perf={:.4f}, med acc={:.4f}'.format(
        avg_perf, avg_acc, med_perf, med_acc))
    cov = np.cov(perfs_to_plot, accs_to_plot)
    ell = get_cov_ellipse(cov, (avg_perf, avg_acc), NSTD)
    ax.add_artist(ell)
    ell.set_clip_box(ax.bbox)
    ell.set_facecolor('r')
    ell.set_alpha(0.5)

    # with open('vs_ellipse.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     print(ell.center, ell.width, ell.height, ell.angle)
    #     writer.writerow([ell.center[0], ell.center[1],
    #                      ell.width, ell.height, ell.angle])
    with open('vs_ellipse.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([ell.center[0], ell.center[1],
                         ell.width, ell.height, ell.angle, med_perf, med_acc])

    with open('vs_perf_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for perf, acc in zip(perfs_to_plot, accs_to_plot):
            writer.writerow([perf, acc])

    # plot cdf
    plt.subplot(2, 1, 2)
    # plt.figure()
    perfs_to_plot = [perf for perf, acc in zip(
        perfs_to_plot, accs_to_plot) if 0.90 <= acc <= 0.95]
    bins, cdf = plot_cdf(perfs_to_plot, 1000, '')
    plt.ylabel('CDF')
    plt.xlabel('Compute Cost(GPU)')

    with open('vs_cdf.csv', 'w') as f:
        writer = csv.writer(f)
        for perf, acc in zip(bins, cdf):
            writer.writerow([perf, acc])


def plot_glimpse_old():
    """Plot Glimpse results."""
    plt.figure()
    plt.subplot(2, 1, 1)
    ax = plt.gca()
    # gl_path = '/home/zxxia/Projects/benchmarking/glimpse/'\
    #     'glimpse_e2e_perfect_tracking_results/glimpse_perfect_tracking_{}.csv'
    gl_path = '/home/zxxia/Projects/benchmarking/glimpse/'\
        'glimpse_e2e_tracking_results/glimpse_tracking_{}.csv'
    # gl_path = '/data/zxxia/benchmarking/glimpse/'\
    #     'e2e_results/glimpse_tracking_{}.csv'
    videos_to_plot = []
    perfs_to_plot = []
    perfs_to_plot = []
    accs_to_plot = []
    for video in VIDEOS:
        filename = gl_path.format(video)
        videos, perfs, accs = load_glimpse_results(filename)
        perfs_to_plot.extend(perfs)
        accs_to_plot.extend(accs)
        videos_to_plot.extend(videos)
        # print(videos)

    plt.title('Glimpse')
    plt.scatter(perfs_to_plot, accs_to_plot, s=PS)
    plt.xlabel('GPU Processing time')
    plt.ylabel('Accuracy')
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    avg_perf = np.mean(perfs_to_plot)
    std_perf = np.std(perfs_to_plot)
    med_perf = np.median(perfs_to_plot)

    avg_acc = np.mean(accs_to_plot)
    std_acc = np.std(accs_to_plot)
    med_acc = np.median(accs_to_plot)

    print('gl avg perf={:.4f}, avg acc={:.4f}, med perf={:.4f}, med acc={:.4f}'.format(
        avg_perf, avg_acc, med_perf, med_acc))
    cov = np.cov(perfs_to_plot, accs_to_plot)
    ell = get_cov_ellipse(cov, (avg_perf, avg_acc), NSTD)
    # if e_gl.height < 0.02:
    #     e_gl.height = 0.02

    ax.add_artist(ell)
    ell.set_clip_box(ax.bbox)
    ell.set_facecolor('r')
    ell.set_alpha(0.5)

    # with open('gl_ellipse.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([ell.center[0], ell.center[1],
    #                      ell.width, ell.height, ell.angle, med_perf, med_acc])
    # with open('gl_perf_acc.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for perf, acc in zip(perfs_to_plot, accs_to_plot):
    #         writer.writerow([perf, acc])
    plt.subplot(2, 1, 2)

    perfs_to_plot = [perf for perf, acc in zip(
        perfs_to_plot, accs_to_plot) if 0.90 <= acc <= 0.95]

    bins, cdf = plot_cdf(perfs_to_plot, 1000, '')
    plt.ylabel('CDF')
    plt.xlabel('Compute Cost(GPU)')

    # plot cdf
    # with open('gl_cdf.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for perf, acc in zip(bins, cdf):
    #         writer.writerow([perf, acc])
    plt.show()


def plot_glimpse():
    """Plot Glimpse results."""
    plt.figure()
    plt.subplot(2, 1, 1)
    ax = plt.gca()
    gl_path = '/data/zxxia/benchmarking/glimpse/'\
        'e2e_results_kcf/glimpse_result_{}.csv'
    results = []
    for video in VIDEOS:
        results.append(pd.read_csv(gl_path.format(video)))
    results = pd.concat(results, ignore_index=True)

    plt.title('Glimpse')
    plt.scatter(results['frame rate'], results['f1'], s=PS)
    plt.xlabel('GPU Processing time')
    plt.ylabel('Accuracy')
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    avg_perf = np.mean(results['frame rate'])
    std_perf = np.std(results['frame rate'])
    med_perf = np.median(results['frame rate'])

    avg_acc = np.mean(results['f1'])
    std_acc = np.std(results['f1'])
    med_acc = np.median(results['f1'])

    print('gl avg perf={:.4f}, avg acc={:.4f}, med perf={:.4f}, med acc={:.4f}'.format(
        avg_perf, avg_acc, med_perf, med_acc))
    cov = np.cov(results['frame rate'], results['f1'])
    ell = get_cov_ellipse(cov, (avg_perf, avg_acc), NSTD)
    # if e_gl.height < 0.02:
    #     e_gl.height = 0.02

    ax.add_artist(ell)
    ell.set_clip_box(ax.bbox)
    ell.set_facecolor('r')
    ell.set_alpha(0.5)

    with open('gl_ellipse.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([ell.center[0], ell.center[1],
                         ell.width, ell.height, ell.angle, med_perf, med_acc])
    with open('gl_perf_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for perf, acc in zip(results['frame rate'], results['f1']):
            writer.writerow([perf, acc])
    plt.subplot(2, 1, 2)

    acc_mask = results['f1'].between(0.90, 0.95)
    bins, cdf = plot_cdf(results['frame rate'].loc[acc_mask], 10000, '')

    plt.ylabel('CDF')
    plt.xlabel('Compute Cost(GPU)')

    # plot cdf
    with open('gl_cdf.csv', 'w') as f:
        writer = csv.writer(f)
        for perf, acc in zip(bins, cdf):
            writer.writerow([perf, acc])
    plt.show()


def plot_awstream():
    """Plot awsream results."""
    plt.figure()
    plt.subplot(2, 1, 1)
    ax = plt.gca()
    aw_path = 'e2e_results_30s_10s/awstream_e2e_results_{}.csv'
    videos_to_plot = []
    bw_to_plot = []
    accs_to_plot = []
    for video in VIDEOS:
        filename = aw_path.format(video)
        videos, bw, accs = load_awstream_results(filename)
        bw_to_plot.extend(bw)
        accs_to_plot.extend(accs)
        videos_to_plot.extend(videos)
        # print(videos)

    plt.title('AWStream')
    plt.scatter(bw_to_plot, accs_to_plot, s=PS)
    plt.xlabel('Network Bandwidth')
    plt.ylabel('Accuracy')
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    # for perf, acc, vid in zip(bw_to_plot, accs_to_plot, videos_to_plot):
    #     ax.annotate(vid, (perf, acc))
    avg_perf = np.mean(bw_to_plot)
    std_perf = np.std(bw_to_plot)
    med_perf = np.median(bw_to_plot)

    avg_acc = np.mean(accs_to_plot)
    std_acc = np.std(accs_to_plot)
    med_acc = np.median(accs_to_plot)

    print('aw avg perf={:.4f}, avg acc={:.4f}, med perf={:.4f}, med acc={:.4f}'.format(
        avg_perf, avg_acc, med_perf, med_acc))
    cov = np.cov(bw_to_plot, accs_to_plot)
    ell = get_cov_ellipse(cov, (avg_perf, avg_acc), NSTD)
    # if e_gl.height < 0.02:
    #     e_gl.height = 0.02

    ax.add_artist(ell)
    ell.set_clip_box(ax.bbox)
    ell.set_facecolor('r')
    ell.set_alpha(0.5)
    with open('aw_ellipse.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([ell.center[0], ell.center[1],
                         ell.width, ell.height, ell.angle, med_perf, med_acc])

    with open('aw_perf_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for perf, acc in zip(bw_to_plot, accs_to_plot):
            writer.writerow([perf, acc])

    plt.subplot(2, 1, 2)
    # plt.figure()
    bw_to_plot = [bw for bw, acc in zip(
        bw_to_plot, accs_to_plot) if 0.90 <= acc <= 0.95]
    bins, cdf = plot_cdf(bw_to_plot, 1000, '')
    plt.ylabel('CDF')
    plt.xlabel('Network Bandwidth')

    # plot cdf
    with open('aw_cdf.csv', 'w') as f:
        writer = csv.writer(f)
        for perf, acc in zip(bins, cdf):
            writer.writerow([perf, acc])


# Plot noscope old results
def plot_noscope_old():
    VIDEOS = ['cropped_crossroad4',  # 'cropped_crossroad4_2',
              'cropped_crossroad5', 'cropped_driving2']
    plt.figure()
    plt.subplot(2, 1, 1)
    ax = plt.gca()
    aw_path = '/home/zxxia/Projects/benchmarking/noscope/noscope_result_{}.csv'
    videos_to_plot = []
    perf_to_plot = []
    accs_to_plot = []
    for video in VIDEOS:
        filename = aw_path.format(video)
        videos, perf, accs = load_noscope_results(filename)
        perf_to_plot.extend(perf)
        accs_to_plot.extend(accs)
        videos_to_plot.extend(videos)
        # print(videos)

    plt.title('Noscope')
    plt.scatter(perf_to_plot, accs_to_plot, s=PS)
    plt.xlabel('Compute Cost(GPU)')
    plt.ylabel('Accuracy')
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    # for perf, acc, vid in zip(perf_to_plot, accs_to_plot, videos_to_plot):
    #     ax.annotate(vid, (perf, acc))

    avg_perf = np.mean(perf_to_plot)
    std_perf = np.std(perf_to_plot)
    med_perf = np.median(perf_to_plot)

    avg_acc = np.mean(accs_to_plot)
    std_acc = np.std(accs_to_plot)
    med_acc = np.median(accs_to_plot)
    print('no avg perf={:.4f}, avg acc={:.4f}, med perf={:.4f}, med acc={:.4f}'.format(
        avg_perf, avg_acc, med_perf, med_acc))

    cov = np.cov(perf_to_plot, accs_to_plot)
    ell = get_cov_ellipse(cov, (avg_perf, avg_acc), NSTD)
    # if e_gl.height < 0.02:
    #     e_gl.height = 0.02

    ax.add_artist(ell)
    ell.set_clip_box(ax.bbox)
    ell.set_facecolor('r')
    ell.set_alpha(0.5)
    with open('no_ellipse.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([ell.center[0], ell.center[1],
                         ell.width, ell.height, ell.angle, med_perf, med_acc])

    with open('no_perf_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for perf, acc in zip(perf_to_plot, accs_to_plot):
            writer.writerow([perf, acc])
    plt.subplot(2, 1, 2)
    perf_to_plot = [perf for perf, acc in zip(
        perf_to_plot, accs_to_plot) if 0.90 <= acc <= 0.95]
    bins, cdf = plot_cdf(perf_to_plot, 1000, '')
    plt.ylabel('CDF')
    plt.xlabel('gpu')

    # plot cdf
    with open('no_cdf.csv', 'w') as f:
        writer = csv.writer(f)
        for perf, acc in zip(bins, cdf):
            writer.writerow([perf, acc])
    plt.show()


def plot_noscope():
    """Plot noscope variance over multiple videos and the compute cost cdf."""
    # results = pd.read_csv(
    #     '../noscope/Noscope_e2e_result_with_frame_diff_allvideo_profile_once_w_gpu_cost_min_gpu.csv')
    results = pd.read_csv(
        '../noscope/Noscope_e2e_result.csv')
    plt.subplot(2, 1, 1)
    ax = plt.gca()
    plt.scatter(results['gpu'], results['f1'])

    avg_perf = np.mean(results['gpu'])
    std_perf = np.std(results['gpu'])
    med_perf = np.median(results['gpu'])

    avg_acc = np.mean(results['f1'])
    std_acc = np.std(results['f1'])
    med_acc = np.median(results['f1'])
    print('no avg perf={:.4f}, avg acc={:.4f}, med perf={:.4f}, med acc={:.4f}'.format(
        avg_perf, avg_acc, med_perf, med_acc))

    cov = np.cov(results['gpu'], results['f1'])
    ell = get_cov_ellipse(cov, (avg_perf, avg_acc), NSTD)
    # if e_gl.height < 0.02:
    #     e_gl.height = 0.02

    ax.add_artist(ell)
    ell.set_clip_box(ax.bbox)
    ell.set_facecolor('r')
    ell.set_alpha(0.5)
    plt.ylabel('f1')
    plt.xlabel('gpu')
    # with open('no_ellipse.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([ell.center[0], ell.center[1],
    #                      ell.width, ell.height, ell.angle, med_perf, med_acc])
    # with open('no_perf_acc.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for perf, acc in zip(results['gpu'], results['f1']):
    #         writer.writerow([perf, acc])
    plt.subplot(2, 1, 2)
    # perf_to_plot = [perf for perf, acc in zip(
    #     perf_to_plot, accs_to_plot) if 0.90 <= acc <= 0.95]
    acc_mask = results['f1'].between(0.90, 0.95)
    bins, cdf = plot_cdf(results['gpu'].loc[acc_mask], 1000, '')
    plt.ylabel('CDF')
    plt.xlabel('gpu')
    plt.axvline(x=0.9, linestyle='--')
    plt.axhline(y=0.5, linestyle='--')
    # with open('no_cdf.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for perf, acc in zip(bins, cdf):
    #         writer.writerow([perf, acc])
    plt.show()
    # pdb.set_trace()


def main():
    plot_noscope()
    # plot_glimpse()


if __name__ == '__main__':
    main()
