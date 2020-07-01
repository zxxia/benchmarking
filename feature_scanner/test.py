import numpy as np
import matplotlib.pyplot as plt
from helpers import load_glimpse_results, load_videostorm_results


def plot_cdf(data, num_bins, title, legend, xlabel, xlim=None):
    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=num_bins, density=True)
    dx = bin_edges[1] - bin_edges[0]
    # Now find the cdf
    cdf = np.cumsum(counts) * dx
    # And finally plot the cdf
    plt.plot(bin_edges[1:], cdf, label=legend)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.ylim([0, 1.1])
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()


# Plot videostorm performance distribution
plt.figure(0)
vs_video_list, perf_list, f1_list = \
    load_videostorm_results('videostorm_t_crossroad.csv')
plot_cdf(perf_list, 1000, 'Videostorm Performance Distribution', 'overfitting',
         'Relative GPU Processing Time')
# plt.axvline(x=0.2852, color='k', linestyle='--')
plt.axhline(y=0.34, color='k', linestyle='--')
vs_video_list, perf_list, f1_list = \
    load_videostorm_results('videostorm_t_crossroad_baseline.csv')
plot_cdf(perf_list, 10000, 't_crossroad: Videostorm Performance Distribution',
         'baseline', 'Relative GPU Processing Time')

# Plot videostorm performance distribution
plt.figure(1)
gl_video_list, perf_list, f1_list = \
    load_glimpse_results('glimpse_t_crossroad.csv')
plot_cdf(perf_list, 10000, 't_crossroad: Glimpse Performance Distribution',
         'overfitting', 'Relative GPU Processing Time')


# Plot videostorm performance vs time
plt.figure(2)
vs_video_list, perf_list, f1_list = \
    load_videostorm_results('videostorm_driving1_3s.csv')
interested_perf_list = []
interested_acc_list = []
for video, perf, acc in zip(vs_video_list, perf_list, f1_list):
    if acc >= 0.87 and acc <= 0.93:
        interested_perf_list.append(perf)
        interested_acc_list.append(acc)

plt.plot(interested_perf_list, 'o-', markersize='3')
plt.plot(interested_acc_list, 'o-', markersize='3')
plt.xlabel('30s videos')
plt.xlabel('3s videos')
plt.ylabel('VideoStorm Relative GPU Processing Time')
plt.title('VideoStorm Relative GPU Processing Time Variation Over Time')

# ft_video_list = list()
# velo_30s = list()
# with open('features_t_crossroad_30s.csv', 'r') as f:
#     f.readline()
#     for line in f:
#         cols = line.strip().split(',')
#         ft_video_list.append(cols[0])
#         velo_30s.append(float(cols[7]))
# plt.figure(1)
# plt.scatter(velo_30s, perf_list, label='t_crossroad')

# velos = list()
# gpu_list = list()
# with open('../results/fig6/vs_obj_velo_data.csv', 'r') as f:
#     f.readline()
#     for line in f:
#         cols = line.strip().split(',')
#         velos.append(float(cols[1]))
#         gpu_list.append(float(cols[2]))

# plt.scatter(velos, gpu_list, label='Selected Videos')
# plt.legend()
# plt.xlim([1,2])
# plt.ylim([0,1.1])
# plt.xlabel('object velocity(median) within 30s')
# plt.ylabel('Relative GPU Processing Time')




# plt.figure(2)
# # Use the histogram function to bin the data
# counts, bin_edges = np.histogram(velo_30s, bins=20, density=True)
# dx = bin_edges[1] - bin_edges[0]
# # Now find the cdf
# cdf = np.cumsum(counts) * dx
# # And finally plot the cdf
# plt.plot(bin_edges[1:], cdf)
# #  plt.title(title)
# plt.xlabel('object velocity(median) within 30s')
# plt.ylabel('CDF')
# plt.ylim([0, 1.1])
# plt.show()
# for name, velo, perf in zip(ft_video_list, velo_30s, perf_list):
#     if velo >= 1.19 and velo <= 1.21:
#         print(name, velo, perf)
plt.show()
