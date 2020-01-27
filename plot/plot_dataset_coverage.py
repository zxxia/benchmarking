import csv
import numpy as np
import matplotlib.pyplot as plt
from benchmarking.videostorm.VideoStorm import load_videostorm_e2e_results
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
vs_path = '/home/zxxia/Projects/benchmarking/videostorm/test_coverage_results/' \
    'videostorm_coverage_results_{}.csv'
for video in VIDEOS:
    filename = vs_path.format(video)
    videos, models, perfs, frame_rates, accs = load_videostorm_e2e_results(
        filename)
    videos_to_plot.extend(videos)
    perfs_to_plot.extend(perfs)
    accs_to_plot.extend(accs)
    if video in CAMERA_TYPES['static']:
        static_perf.extend(perfs)
        static_acc.extend(accs)
    elif video in CAMERA_TYPES['moving']:
        moving_perf.extend(perfs)
        moving_acc.extend(accs)

# plt.scatter(static_perf, static_acc, s=PS, label='static')
# plt.scatter(moving_perf, moving_acc, s=PS, label='moving')
# plt.legend()
# plt.title('videostorm on static vs moving ')

# plt.figure()
print(len(videos_to_plot))
videos, models, perfs, frame_rates, accs = load_videostorm_e2e_results(
    '../videostorm/kitti_e2e/videostorm_e2e_kitti.csv')
videos_to_plot.extend(videos)
perfs_to_plot.extend(perfs)
accs_to_plot.extend(accs)
# with open('kitti_coverage_vs.csv', 'w', 1) as f:
#     writer = csv.writer(f)
#     writer.writerow(['cost', 'acc'])
#     for perf, acc in zip(perfs, accs):
#         writer.writerow([perf, acc])

plt.xlabel('GPU Processing time')
plt.ylabel('Accuracy')
plt.title('Our 6.21hr KITTI: 0.34hr')

plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
plt.scatter(perfs_to_plot, accs_to_plot, s=PS, label='our')
plt.scatter(perfs, accs, s=PS, label='kitti')
plt.legend()

plt.figure()
videos, models, perfs, frame_rates, accs = load_videostorm_e2e_results(
    '../videostorm/waymo_720p_e2e/videostorm_e2e_waymo.csv')
perfs_to_plot.extend(perfs)
accs_to_plot.extend(accs)
plt.scatter(perfs_to_plot, accs_to_plot, s=PS, label='our')
plt.scatter(perfs, accs, s=PS, label='waymo')
plt.legend()
plt.xlabel('GPU Processing time')
plt.ylabel('Accuracy')
plt.title('Our 6.21hr Waymo: 5.56hr')
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)

plt.figure()
videos, models, perfs, frame_rates, accs = load_videostorm_e2e_results(
    '../videostorm/canada_e2e/videostorm_e2e_canada.csv')
perfs_to_plot.extend(perfs)
accs_to_plot.extend(accs)
plt.scatter(perfs_to_plot, accs_to_plot, s=PS, label='our')
plt.scatter(perfs, accs, s=PS, label='Canada Crossroad')
plt.legend()
plt.xlabel('GPU Processing time')
plt.ylabel('Accuracy')
plt.title('Our 6.21hr Canada Crossroad: {}hr'.format(len(videos)*30/3600))
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
# with open('canada_coverage_vs.csv', 'w', 1) as f:
#     writer = csv.writer(f)
#     writer.writerow(['cost', 'acc'])
#     for perf, acc in zip(perfs, accs):
#         writer.writerow([perf, acc])

plt.figure()
videos, models, perfs, frame_rates, accs = load_videostorm_e2e_results(
    '../videostorm/test_coverage_results/videostorm_coverage_results_t_crossroad.csv')
perfs_to_plot.extend(perfs)
accs_to_plot.extend(accs)
plt.scatter(perfs_to_plot, accs_to_plot, s=PS, label='our')
plt.scatter(perfs, accs, s=PS, label='t_crossroad')
plt.legend()
plt.xlabel('GPU Processing time')
plt.ylabel('Accuracy')
plt.title('Our 6.21hr t_crossroad: 2.4hr')
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
# with open('t_crossroad_coverage_vs.csv', 'w', 1) as f:
#     writer = csv.writer(f)
#     writer.writerow(['cost', 'acc'])
#     for perf, acc in zip(perfs, accs):
#         writer.writerow([perf, acc])

plt.figure()
videos, models, perfs, frame_rates, accs = load_videostorm_e2e_results(
    '../videostorm/test_coverage_results/videostorm_coverage_results_road_trip.csv')
perfs_to_plot.extend(perfs)
accs_to_plot.extend(accs)
plt.scatter(perfs_to_plot, accs_to_plot, s=PS, label='our')
plt.scatter(perfs, accs, s=PS, label='road_trip')
plt.legend()
plt.xlabel('GPU Processing time')
plt.ylabel('Accuracy')
plt.title('Our 6.21hr road trip: 5.5hr')
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)

# with open('our_dataset_coverage_vs.csv', 'w', 1) as f:
#     writer = csv.writer(f)
#     writer.writerow(['cost', 'acc'])
#     for perf, acc in zip(perfs_to_plot, accs_to_plot):
#         writer.writerow([perf, acc])

plt.figure()
videos, models, perfs, frame_rates, accs = load_videostorm_e2e_results(
    '../videostorm/mot16_e2e/videostorm_e2e_mot16.csv')
videos_to_plot.extend(videos)
perfs_to_plot.extend(perfs)
accs_to_plot.extend(accs)
with open('mot16_coverage_vs.csv', 'w', 1) as f:
    writer = csv.writer(f)
    writer.writerow(['cost', 'acc'])
    for perf, acc in zip(perfs, accs):
        writer.writerow([perf, acc])

plt.scatter(perfs_to_plot, accs_to_plot, s=PS, label='our')
plt.scatter(perfs, accs, s=PS, label='mot16')
plt.xlabel('GPU Processing time')
plt.ylabel('Accuracy')
plt.title('Our 6.21hr mot16: 0.34hr')
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
plt.show()
