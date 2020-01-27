import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from benchmarking.constants import RESOL_DICT

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

results = pd.read_csv('../feature_analysis/temporal_scan_error.csv')
resolution_list = [resol for resol in results['sample_step'].to_list()]
resolution_list = 1/np.array(resolution_list)
print(resolution_list)
plt.figure()
mask = [False, False, True, False, True, True, True]
plt.errorbar(resolution_list[mask],
             results['estimation error mean'][mask].to_list(),
             yerr=results['estimation error std'][mask].to_list(),
             fmt='o-', label='scan')
plt.errorbar(resolution_list[mask]*3,
             results['baseline estimation error mean'][mask].to_list(),
             yerr=results['baseline estimation error std'][mask].to_list(),
             fmt='o-', label='baseline')
plt.title('Good Case Percent')
plt.legend()
plt.xlabel('gpu cycle(normalized)')
plt.ylabel('estimation error')
plt.ylim(0, 1)
plt.xlim(0, 0.15)

plt.figure()
plt.errorbar(resolution_list[mask],
             results['bad estimation error mean'][mask].to_list(),
             yerr=results['bad estimation error std'][mask].to_list(),
             fmt='o-', label='scan')
plt.errorbar(resolution_list[mask]*3,
             results['bad baseline estimation error mean'][mask].to_list(),
             yerr=results['bad baseline estimation error std'][mask].to_list(),
             fmt='o-', label='baseline')
plt.title('Bad Case Percent')
plt.legend()
plt.xlabel('gpu cycle(normalized)')
plt.ylabel('estimation error')
plt.xlim(0, 0.15)
plt.ylim(0, 1)
with open('vs_feature_scan.csv', 'w') as f:
    writer = csv.writer(f)
    # for resol, est_err_mean, est_err_std, baseline_est_err_mean, baseline_est_err_std in \
    writer.writerow(['scan gpu', 'estimation error mean',
                     'estimation error std',
                     'bad estimation error mean',
                     'bad estimation error std',
                     'baseline gpu', 'baseline estimation error mean',
                     'baseline estimation error std',
                     'bad baseline estimation error mean',
                     'bad baseline estimation error std',

                     ])

    writer.writerows(zip(resolution_list[mask],
                         results['estimation error mean'][mask].to_list(),
                         results['estimation error std'][mask].to_list(),
                         results['bad estimation error mean'][mask].to_list(),
                         results['bad estimation error std'][mask].to_list(),
                         resolution_list[mask]*3,
                         results['baseline estimation error mean'][mask].to_list(),
                         results['baseline estimation error std'][mask].to_list(),
                         results['bad baseline estimation error mean'][mask].to_list(),
                         results['bad baseline estimation error std'][mask].to_list(),

                         ))
plt.show()
