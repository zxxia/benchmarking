import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4', 'drift',
          'driving1',  'driving_downtown', 'highway',
          'nyc', 'jp',  'lane_split', 'driving2',
          'motorway', 'park', 'russia', 'russia1', 'traffic', 'tw', 'tw1',
          'tw_under_bridge']
VIDEOS = ['russia1']  # 'driving1',
PS = 2

# vs_path = '/data/zxxia/benchmarking/videostorm/test_coverage_results/' \
#     'videostorm_coverage_results_{}.csv'
vs_path = '/data/zxxia/benchmarking/results/videostorm/'\
    'test_coverage_results/videostorm_coverage_results_{}.csv'
# gl_path = '/home/zxxia/Projects/benchmarking/glimpse/'\
#     'glimpse_e2e_tracking_results/glimpse_tracking_{}.csv'
gl_path = '/data/zxxia/benchmarking/glimpse/'\
    'e2e_results_kcf/glimpse_result_{}.csv'
for video in VIDEOS:
    vs_results = pd.read_csv(vs_path.format(video))
    # plt.scatter(np.arange(len(vs_results)), vs_results['f1'])
    # plt.show()
    gl_results = pd.read_csv(gl_path.format(video))
    assert len(vs_results) == len(gl_results)
    ts = np.arange(len(vs_results))*30
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(ts, vs_results['f1'], 'o-', ms=PS, label='VideoStorm')
    plt.plot(ts, gl_results['f1'], 'o-', ms=PS, label='Glimpse')
    plt.xlabel('Time(s)')
    plt.ylabel('Accuracy')
    plt.title(video)

    plt.ylim(0, 1.1)
    ax = plt.gca()
    for t, gl_perf, vs_perf, gl_acc, vs_acc, vid in zip(
            ts, gl_results['frame rate'], vs_results['gpu time'],
            gl_results['f1'], vs_results['f1'], vs_results['video_name']):
        if vs_perf == 0.0 or vs_acc == 0.0:
            continue
        if gl_perf / vs_perf < 1.0 and gl_acc / vs_acc > 1.0:
            ax.annotate(vid, (t, gl_acc))
        if gl_perf / vs_perf > 1.0 and gl_acc / vs_acc < 1.0:
            ax.annotate(vid, (t, vs_acc))

    plt.subplot(2, 1, 2)
    ax = plt.gca()

    plt.plot(ts, vs_results['gpu time'], 'o-', ms=PS, label='VideoStorm')
    plt.plot(ts, gl_results['frame rate'], 'o-', ms=PS, label='Glimpse')
    plt.ylim(0, 1.1)
    plt.xlabel('Time(s)')
    plt.ylabel('Cost')

    for t, gl_perf, vs_perf, gl_acc, vs_acc, vid in zip(
            ts, gl_results['frame rate'], vs_results['gpu time'],
            gl_results['f1'], vs_results['f1'], vs_results['video_name']):
        if vs_perf == 0.0 or vs_acc == 0.0:
            continue
        if gl_perf / vs_perf < 1.0 and gl_acc / vs_acc > 1.0:
            ax.annotate(vid, (t, gl_perf))
        if gl_perf / vs_perf > 1.0 and gl_acc / vs_acc < 1.0:
            ax.annotate(vid, (t, vs_perf))
    plt.tight_layout()
    plt.legend()
    # with open('var_over_time.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for t, vs_acc, vs_perf, gl_acc, gl_perf in zip(
    #             ts, vs_results['f1'], vs_results['gpu time'],
    #             gl_results['f1'], gl_results['frame rate']):
    #         writer.writerow([t, vs_acc, vs_perf, gl_acc, gl_perf])
    plt.show()
