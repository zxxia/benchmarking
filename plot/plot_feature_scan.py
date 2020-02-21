import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from benchmarking.constants import RESOL_DICT
from collections import defaultdict
import pdb

# results = pd.read_csv('../feature_analysis/spatial_scan_error_480p_best.csv')
# resolution_list = [resol for resol in results['resolution'].to_list()]
# resolution_list = 1/np.array(resolution_list)
# print(resolution_list)
# plt.figure()
# mask = [True, True, False, False, True, False, True]
# plt.errorbar(resolution_list[mask],
#              results['estimation error mean'][mask].to_list(),
#              yerr=results['estimation error std'][mask].to_list(),
#              fmt='o-', label='scan')
# plt.errorbar(resolution_list[-4:]*3,
#              results['baseline estimation error mean'].to_list()[-4:],
#              yerr=results['baseline estimation error std'].to_list()[-4:],
#              fmt='o-', label='baseline')
# plt.title('Good Case Percent')
# plt.legend()
# plt.xlabel('gpu cycle(normalized)')
# plt.ylabel('estimation error')
# plt.ylim(0, 1)
# plt.xlim(0, 0.15)
#
# plt.figure()
# plt.errorbar(resolution_list[mask],
#              results['bad estimation error mean'][mask].to_list(),
#              yerr=results['bad estimation error std'][mask].to_list(),
#              fmt='o-', label='scan')
# plt.errorbar(resolution_list[-4:]*3,
#              results['bad baseline estimation error mean'].to_list(
# )[-4:], yerr=results['bad baseline estimation error std'].to_list()[-4:], fmt='o-', label='baseline')
# plt.title('Bad Case Percent')
# plt.legend()
# plt.xlabel('gpu cycle(normalized)')
# plt.ylabel('estimation error')
# plt.xlim(0, 0.15)
# plt.ylim(0, 1)
#
# with open('aws_feature_scan.csv', 'w') as f:
#     writer = csv.writer(f)
#     # for resol, est_err_mean, est_err_std, baseline_est_err_mean, baseline_est_err_std in \
#     writer.writerow(['scan gpu', 'estimation error mean',
#                      'estimation error std',
#                      'bad estimation error mean',
#                      'bad estimation error std',
#                      'baseline gpu', 'baseline estimation error mean',
#                      'baseline estimation error std',
#                      'bad baseline estimation error mean',
#                      'bad baseline estimation error std',
#
#                      ])
#
#     writer.writerows(zip(resolution_list[mask],
#                          results['estimation error mean'][mask].to_list(),
#                          results['estimation error std'][mask].to_list(),
#                          results['bad estimation error mean'][mask].to_list(),
#                          results['bad estimation error std'][mask].to_list(),
#                          resolution_list[-4:]*3,
#                          results['baseline estimation error mean'][-4:].to_list(),
#                          results['baseline estimation error std'][-4:].to_list(),
#                          results['bad baseline estimation error mean'][-4:].to_list(),
#                          results['bad baseline estimation error std'][-4:].to_list(),
#
#                          ))

# results = pd.read_csv('../feature_analysis/temporal_scan_error_new.csv')
# resolution_list = [resol for resol in results['sample_step'].to_list()]
# resolution_list = 1/np.array(resolution_list)
# print(resolution_list)
# plt.figure()
# # mask = [False, False, True, False, True, True, True]
# mask = [False, True, True, True, True, True, False]
# plt.errorbar(resolution_list[mask],
#              results['estimation error mean'][mask].to_list(),
#              yerr=results['estimation error std'][mask].to_list(),
#              fmt='o-', label='scan')
# plt.errorbar(resolution_list[mask]*3,
#              results['baseline estimation error mean'][mask].to_list(),
#              yerr=results['baseline estimation error std'][mask].to_list(),
#              fmt='o-', label='baseline')
# plt.title('Good Case Percent')
# plt.legend()
# plt.xlabel('gpu cycle(normalized)')
# plt.ylabel('estimation error')
# plt.ylim(0, 1)
# plt.xlim(0, 0.15)
#
# plt.figure()
# plt.errorbar(resolution_list[mask],
#              results['bad estimation error mean'][mask].to_list(),
#              yerr=results['bad estimation error std'][mask].to_list(),
#              fmt='o-', label='scan')
# plt.errorbar(resolution_list[mask]*3,
#              results['bad baseline estimation error mean'][mask].to_list(),
#              yerr=results['bad baseline estimation error std'][mask].to_list(),
#              fmt='o-', label='baseline')
# plt.title('Bad Case Percent')
# plt.legend()
# plt.xlabel('gpu cycle(normalized)')
# plt.ylabel('estimation error')
# plt.xlim(0, 0.15)
# plt.ylim(0, 1)
# with open('vs_feature_scan.csv', 'w') as f:
#     writer = csv.writer(f)
#     # for resol, est_err_mean, est_err_std, baseline_est_err_mean, baseline_est_err_std in \
#     writer.writerow(['scan gpu', 'estimation error mean',
#                      'estimation error std',
#                      'bad estimation error mean',
#                      'bad estimation error std',
#                      'baseline gpu', 'baseline estimation error mean',
#                      'baseline estimation error std',
#                      'bad baseline estimation error mean',
#                      'bad baseline estimation error std',
#
#                      ])
#
#     writer.writerows(zip(resolution_list[mask],
#                          results['estimation error mean'][mask].to_list(),
#                          results['estimation error std'][mask].to_list(),
#                          results['bad estimation error mean'][mask].to_list(),
#                          results['bad estimation error std'][mask].to_list(),
#                          resolution_list[mask]*3,
#                          results['baseline estimation error mean'][mask].to_list(),
#                          results['baseline estimation error std'][mask].to_list(),
#                          results['bad baseline estimation error mean'][mask].to_list(),
#                          results['bad baseline estimation error std'][mask].to_list(),
#
#                          ))


plt.figure()

sample_step_estimation_errors = defaultdict(list)
sample_step_baseline_estimation_errors = defaultdict(list)
sample_step_bad_estimation_errors = defaultdict(list)
sample_step_bad_baseline_estimation_errors = defaultdict(list)
for i in range(300):
    results = pd.read_csv(
        f'../feature_analysis/round_ft_scan_results/road_trip_scan_error_round{i}.csv')

    for row_idx, row in results.iterrows():
        sample_step_estimation_errors[row['sample_step']].append(
            row['estimation error'])
        sample_step_baseline_estimation_errors[row['sample_step']].append(
            row['baseline estimation error'])
        sample_step_bad_estimation_errors[row['sample_step']].append(
            row['bad estimation error'])
        sample_step_bad_baseline_estimation_errors[row['sample_step']].append(
            row['bad baseline estimation error'])

print(sample_step_bad_baseline_estimation_errors)
sorted_sample_step = sorted(sample_step_estimation_errors.keys())
gpu_norm = 1/np.array(sorted_sample_step)
error_to_plot = np.array([np.mean(sample_step_estimation_errors[key])
                          for key in sorted_sample_step])
error_std_to_plot = np.array([np.std(sample_step_estimation_errors[key])
                              for key in sorted_sample_step])

baseline_error_to_plot = np.array([np.mean(
    sample_step_baseline_estimation_errors[key]) for key in sorted_sample_step])
baseline_error_std_to_plot = np.array([np.std(sample_step_baseline_estimation_errors[key])
                                       for key in sorted_sample_step])


mask = [False, False, True, True, False, True, False, False, True,
        False, True, False, False, False, True, False, True]
baseline_mask = [True, True, True, False, False, True, False, False, False,
                 False, True, True, False, True, True, False, False]
plt.errorbar(np.log10(gpu_norm[mask]/3), error_to_plot[mask],  # yerr=error_std_to_plot[mask],
             fmt='o-', label='scan')
plt.errorbar(np.log10(gpu_norm[baseline_mask]), baseline_error_to_plot[baseline_mask],
             fmt='o-', label='baseline')

bad_error_to_plot = np.array([np.mean(sample_step_bad_estimation_errors[key])
                              for key in sorted_sample_step])
bad_error_std_to_plot = np.array([np.std(sample_step_bad_estimation_errors[key])
                                  for key in sorted_sample_step])
plt.title('videostorm good cases')
plt.xlabel('Normalized gpu')
plt.ylabel('error')
plt.ylim(0, 0.5)
# ax = plt.gca()
# ax.set_xscale('log')


bad_baseline_error_to_plot = np.array([np.mean(
    sample_step_bad_baseline_estimation_errors[key])for key in sorted_sample_step])

bad_baseline_error_std_to_plot = np.array([np.std(sample_step_bad_baseline_estimation_errors[key])
                                           for key in sorted_sample_step])
plt.figure()
bad_mask = [False, False, True, True, True, False, True, False, True,
            False, True, False, False, False, True, False, True]
bad_baseline_mask = [True, True, True, True, False, True, False, False, True,
                     False, True, False, True, False, False, False, False]
print(type(gpu_norm))
print(len(gpu_norm))
plt.errorbar(np.log10(gpu_norm[bad_mask]/3), bad_error_to_plot[bad_mask],
             fmt='o-', label='scan')
plt.errorbar(np.log10(gpu_norm[bad_baseline_mask]), bad_baseline_error_to_plot[bad_baseline_mask],
             fmt='o-', label='baseline')
plt.title('videostorm bad cases')
plt.xlabel('Normalized gpu')
plt.ylabel('error')
plt.legend()
plt.ylim(0, 0.5)

with open('vs_feature_scan_log.csv', 'w', 1) as f:
    writer = csv.writer(f)
    writer.writerow(['scan good case relative gpu',
                     'scan good case percentage error mean',
                     'scan good case percentage error std',
                     'baseline good case relative gpu',
                     'baseline good case percentage error mean',
                     'baseline good case percentage error std',
                     'scan bad case relative gpu',
                     'scan bad case percentage error mean',
                     'scan bad case percentage error std',
                     'baseline bad case relative gpu',
                     'baseline bad case percentage error mean',
                     'baseline bad case percentage error std'])
    writer.writerows(
        zip(np.log10(gpu_norm[mask]/3), error_to_plot[mask],
            error_std_to_plot[mask],
            np.log10(gpu_norm[baseline_mask]),
            baseline_error_to_plot[baseline_mask],
            baseline_error_std_to_plot[baseline_mask],
            np.log10(gpu_norm[bad_mask]/3),
            bad_error_to_plot[bad_mask], bad_error_std_to_plot[bad_mask],
            np.log10(gpu_norm[bad_baseline_mask]),
            bad_baseline_error_to_plot[bad_baseline_mask],
            bad_baseline_error_std_to_plot[bad_baseline_mask]))
# with open('vs_feature_scan_log.csv', 'w', 1) as f:
#     writer = csv.writer(f)
#     writer.writerow(['relative gpu',
#                      'scan good case percentage error mean',
#                      'scan good case percentage error std',
#                      'baseline good case percentage error mean',
#                      'baseline good case percentage error std',
#                      'scan bad case percentage error mean',
#                      'scan bad case percentage error std',
#                      'baseline bad case percentage error mean',
#                      'baseline bad case percentage error std'])
#     writer.writerows(
#         zip(gpu_norm[mask], error_to_plot[mask], error_std_to_plot[mask],
#             baseline_error_to_plot[mask], baseline_error_std_to_plot[mask],
#             bad_error_to_plot[bad_mask], bad_error_std_to_plot[bad_mask],
#             bad_baseline_error_to_plot[bad_mask],
#             bad_baseline_error_std_to_plot[bad_mask]))


plt.figure()

sample_step_estimation_errors = defaultdict(list)
sample_step_baseline_estimation_errors = defaultdict(list)
sample_step_bad_estimation_errors = defaultdict(list)
sample_step_bad_baseline_estimation_errors = defaultdict(list)
for i in range(300):
    # results = pd.read_csv(
    #     f'../feature_analysis/round_ft_scan_results/road_trip_scan_error_round{i}.csv')
    results = pd.read_csv(
        f'../feature_analysis/round_ft_scan_results/road_trip_aws_scan_error_round{i}.csv')

    for row_idx, row in results.iterrows():
        sample_step_estimation_errors[row['sample_step']].append(
            row['estimation error'])
        sample_step_baseline_estimation_errors[row['sample_step']].append(
            row['baseline estimation error'])
        sample_step_bad_estimation_errors[row['sample_step']].append(
            row['bad estimation error'])
        sample_step_bad_baseline_estimation_errors[row['sample_step']].append(
            row['bad baseline estimation error'])

print(sample_step_bad_baseline_estimation_errors)
sorted_sample_step = sorted(sample_step_estimation_errors.keys())
gpu_norm = 1/np.array(sorted_sample_step)
error_to_plot = np.array([np.mean(sample_step_estimation_errors[key])
                          for key in sorted_sample_step])
error_std_to_plot = np.array([np.std(sample_step_estimation_errors[key])
                              for key in sorted_sample_step])

baseline_error_to_plot = np.array([np.mean(
    sample_step_baseline_estimation_errors[key]) for key in sorted_sample_step])
baseline_error_std_to_plot = np.array([np.std(sample_step_baseline_estimation_errors[key])
                                       for key in sorted_sample_step])


# mask = [True, True, True, False, True, False, True,
#         False, True, False, False, False, True, False, True]
mask = [False, False, False, False, True, True, False, False,
        True, True, True, True, False, False, True, True]
baseline_mask = [True, True, True, False, True, False, False, True,
                 False, False, True, False, False, False, True, True]
plt.errorbar(np.log10(gpu_norm[mask]/3), error_to_plot[mask],  # yerr=error_std_to_plot[mask],
             fmt='o-', label='scan')
plt.errorbar(np.log10(gpu_norm[baseline_mask]), baseline_error_to_plot[baseline_mask],
             fmt='o-', label='baseline')

bad_error_to_plot = np.array([np.mean(sample_step_bad_estimation_errors[key])
                              for key in sorted_sample_step])
bad_error_std_to_plot = np.array([np.std(sample_step_bad_estimation_errors[key])
                                  for key in sorted_sample_step])
# plt.title('videostorm good cases')
plt.title('aws good cases')
plt.xlabel('Normalized gpu')
plt.ylabel('error')
plt.ylim(0, 0.5)
# ax = plt.gca()
# ax.set_xscale('log')


bad_baseline_error_to_plot = np.array([np.mean(
    sample_step_bad_baseline_estimation_errors[key])for key in sorted_sample_step])

bad_baseline_error_std_to_plot = np.array([np.std(sample_step_bad_baseline_estimation_errors[key])
                                           for key in sorted_sample_step])
plt.figure()
# bad_mask = [True, True, True, False, True, False, True,
#             False, True, False, False, False, True, False, True]
bad_mask = [False, False, True, True, False, True, True, True,
            False, True, False, True, False, False, True, False]
print(type(gpu_norm))
print(len(gpu_norm))
plt.errorbar(np.log10(gpu_norm[bad_mask]/3), bad_error_to_plot[bad_mask],
             fmt='o-', label='scan')
baseline_bad_mask = [True, True, True, True, False, True, False, False,
                     False, False, True, False, False, False, True, True]
plt.errorbar(np.log10(gpu_norm[baseline_bad_mask]), bad_baseline_error_to_plot[baseline_bad_mask],
             fmt='o-', label='baseline')
# plt.title('videostorm bad cases')
plt.title('aws bad cases')
plt.xlabel('Normalized gpu')
plt.ylabel('error')
plt.legend()
plt.ylim(0, 0.5)

with open('aws_feature_scan_log.csv', 'w', 1) as f:
    writer = csv.writer(f)
    writer.writerow(['scan good case relative gpu',
                     'scan good case percentage error mean',
                     'scan good case percentage error std',
                     'baseline good case relative gpu',
                     'baseline good case percentage error mean',
                     'baseline good case percentage error std',
                     'scan bad case relative gpu',
                     'scan bad case percentage error mean',
                     'scan bad case percentage error std',
                     'baseline bad case relative gpu',
                     'baseline bad case percentage error mean',
                     'baseline bad case percentage error std'])
    writer.writerows(
        zip(np.log10(gpu_norm[mask]/3), error_to_plot[mask], error_std_to_plot[mask],
            np.log10(gpu_norm[baseline_mask]
                     ), baseline_error_to_plot[baseline_mask],
            baseline_error_std_to_plot[baseline_mask],
            np.log10(gpu_norm[bad_mask]/3),
            bad_error_to_plot[bad_mask], bad_error_std_to_plot[bad_mask],
            np.log10(gpu_norm[baseline_bad_mask]),
            bad_baseline_error_to_plot[baseline_bad_mask],
            bad_baseline_error_std_to_plot[baseline_bad_mask]))
# with open('aws_feature_scan_new.csv', 'w', 1) as f:
# writer = csv.writer(f)
# writer.writerow(['scan good case relative gpu',
# 'scan good case percentage error mean',
# 'scan good case percentage error std',
# 'baseline good case relative gpu',
# 'baseline good case percentage error mean',
# 'baseline good case percentage error std',
# 'scan bad case relative gpu',
# 'scan bad case percentage error mean',
# 'scan bad case percentage error std',
# 'baseline bad case relative gpu',
# 'baseline bad case percentage error mean',
# 'baseline bad case percentage error std'])
#     writer.writerows(
#         zip(gpu_norm[mask]/3, error_to_plot[mask], error_std_to_plot[mask],
#             gpu_norm[baseline_mask], baseline_error_to_plot[baseline_mask],
#             baseline_error_std_to_plot[baseline_mask],
#             gpu_norm[bad_mask]/3,
#             bad_error_to_plot[bad_mask], bad_error_std_to_plot[bad_mask],
#             gpu_norm[baseline_bad_mask],
#             bad_baseline_error_to_plot[baseline_bad_mask],
#             bad_baseline_error_std_to_plot[baseline_bad_mask]))
# with open('vs_feature_scan_new.csv', 'w', 1) as f:
#     writer = csv.writer(f)
#     writer.writerow(['relative gpu',
#                      'scan good case percentage error mean',
#                      'scan good case percentage error std',
#                      'baseline good case percentage error mean',
#                      'baseline good case percentage error std',
#                      'scan bad case percentage error mean',
#                      'scan bad case percentage error std',
#                      'baseline bad case percentage error mean',
#                      'baseline bad case percentage error std'])
#     writer.writerows(
#         zip(gpu_norm[mask], error_to_plot[mask], error_std_to_plot[mask],
#             baseline_error_to_plot[mask], baseline_error_std_to_plot[mask],
#             bad_error_to_plot[bad_mask], bad_error_std_to_plot[bad_mask],
#             bad_baseline_error_to_plot[bad_mask],
#             bad_baseline_error_std_to_plot[bad_mask]))

plt.show()
