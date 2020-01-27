import csv
import numpy as np
import matplotlib.pyplot as plt
from benchmarking.glimpse.Glimpse import load_glimpse_results
from benchmarking.constants import CAMERA_TYPES


VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4', 'drift',
          'driving1', 'driving_downtown', 'highway',
          'nyc', 'jp',  'lane_split',  # 'driving2',
          'motorway', 'park', 'russia', 'russia1', 'traffic', 'tw', 'tw1',
          'tw_under_bridge']  #
PS = 2

static_perf = []
moving_perf = []
static_acc = []
moving_acc = []
videos_to_plot = []
perfs_to_plot = []
accs_to_plot = []
# vs_path = '/data/zxxia/benchmarking/videostorm/test_coverage_results/' \
#     'videostorm_coverage_results_{}.csv'
# vs_path = '/home/zxxia/Projects/benchmarking/glimpse/test/glimpse_e2e_tracking_results/glimpse_tracking_{}.csv'
# vs_path = '/home/zxxia/Projects/benchmarking/glimpse/test/glimpse_e2e_perfect_tracking_results/glimpse_perfect_tracking_{}.csv'
vs_path = '/data/zxxia/benchmarking/glimpse/e2e_results/glimpse_tracking_{}.csv'
for video in VIDEOS:
    filename = vs_path.format(video)
    videos, perfs, accs = load_glimpse_results(filename)
    videos_to_plot.extend(videos)
    perfs_to_plot.extend(perfs)
    accs_to_plot.extend(accs)
    if video in CAMERA_TYPES['static']:
        static_perf.extend(perfs)
        static_acc.extend(accs)
    elif video in CAMERA_TYPES['moving']:
        moving_perf.extend(perfs)
        moving_acc.extend(accs)
ax = plt.gca()
plt.scatter(static_perf, static_acc, s=PS, label='static')
plt.scatter(moving_perf, moving_acc, s=PS, label='moving')
for perf, acc, vid in zip(perfs_to_plot, accs_to_plot, videos_to_plot):
    if 'drift' in vid:
        ax.annotate(vid, (perf, acc))
plt.legend()
plt.title('glimpse on static vs moving ')

plt.figure()
print(len(videos_to_plot))
# videos, models, perfs, frame_rates, accs = load_videostorm_e2e_results(
#     '../videostorm/kitti_e2e/videostorm_e2e_kitti.csv')
# videos_to_plot.extend(videos)
# perfs_to_plot.extend(perfs)
# accs_to_plot.extend(accs)

# plt.xlabel('GPU Processing time')
# plt.ylabel('Accuracy')
# plt.title('Our 6.21hr KITTI: 0.34hr')
#
# plt.xlim(0, 1.1)
# plt.ylim(0, 1.1)
# plt.scatter(perfs_to_plot, accs_to_plot, s=PS, label='our')
# plt.scatter(perfs, accs, s=PS, label='kitti')
# plt.legend()
#
# plt.figure()
videos, perfs, accs = load_glimpse_results(
    '../glimpse/test/glimpse_e2e_waymo.csv')
plt.scatter(perfs_to_plot, accs_to_plot, s=PS, label='our')
plt.scatter(perfs, accs, s=PS, label='waymo')
plt.legend()
plt.xlabel('GPU Processing time')
plt.ylabel('Accuracy')
plt.title('Our 6.21hr Waymo: 5.56hr')
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)

# plt.figure()
# videos, models, perfs, frame_rates, accs = load_videostorm_e2e_results(
#     '../videostorm/canada_e2e/videostorm_e2e_canada.csv')
# print()
# plt.scatter(perfs_to_plot, accs_to_plot, s=PS, label='our')
# plt.scatter(perfs, accs, s=PS, label='Canada Crossroad')
# plt.legend()
# plt.xlabel('GPU Processing time')
# plt.ylabel('Accuracy')
# plt.title('Our 6.21hr Canada Crossroad: {}hr'.format(len(videos)*30/3600))
# plt.xlim(0, 1.1)
# plt.ylim(0, 1.1)

# plt.figure()
# videos, models, perfs, frame_rates, accs = load_videostorm_e2e_results(
#     '../videostorm/test_coverage_results/videostorm_coverage_results_t_crossroad.csv')
# plt.scatter(perfs_to_plot, accs_to_plot, s=PS, label='our')
# plt.scatter(perfs, accs, s=PS, label='t_crossroad')
# plt.legend()
# plt.xlabel('GPU Processing time')
# plt.ylabel('Accuracy')
# plt.title('Our 6.21hr t_crossroad: 2.4hr')
# plt.xlim(0, 1.1)
# plt.ylim(0, 1.1)
#
# plt.figure()
# videos, models, perfs, frame_rates, accs = load_videostorm_e2e_results(
#     '../videostorm/test_coverage_results/videostorm_coverage_results_road_trip.csv')
# plt.scatter(perfs_to_plot, accs_to_plot, s=PS, label='our')
# plt.scatter(perfs, accs, s=PS, label='road_trip')
# plt.legend()
# plt.xlabel('GPU Processing time')
# plt.ylabel('Accuracy')
# plt.title('Our 6.21hr road trip: 5.5hr')
# plt.xlim(0, 1.1)
# plt.ylim(0, 1.1)
plt.show()

