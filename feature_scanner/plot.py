import numpy as np
import matplotlib.pyplot as plt

plt.subplot(4, 1, 1)
plt.bar(np.arange(0, 3), [0.173, 0.072, 0.33], width=0.1)
plt.bar(np.arange(3, 6), [0.0, 0.0, 0.0], width=0.1)
plt.xticks(np.arange(6), ['gt good', 'scan good', 'bl good',
                          'gt bad', 'scan bad', 'bl bad'])
plt.title('driving_downtown 30s', fontsize=8)
plt.subplot(4, 1, 2)
plt.bar(np.arange(0, 3), [0.27, 0.22, 0], width=0.1)
plt.bar(np.arange(3, 6), [0.01, 0.02, 0.25], width=0.1)
plt.xticks(np.arange(6), ['gt good', 'scan good', 'bl good',
                          'gt bad', 'scan bad', 'bl bad'])
plt.title('driving_downtown 10s', fontsize=8)

plt.subplot(4, 1, 3)
plt.bar(np.arange(0, 3), [0.31, 0.23, 0.125, ], width=0.1)
plt.bar(np.arange(3, 6), [0.016, 0.021, 0.25], width=0.1)
plt.xticks(np.arange(6), ['gt good', 'scan good', 'bl good',
                          'gt bad', 'scan bad', 'bl bad'])
plt.title('driving_downtown 5s', fontsize=8)
plt.subplot(4, 1, 4)
plt.bar(np.arange(0, 3), [0.15, 0.29, 0.57, ], width=0.1)
plt.bar(np.arange(3, 6), [0.017, 0.04, 0.0], width=0.1)
plt.xticks(np.arange(6), ['gt good', 'scan good', 'bl good',
                          'gt bad', 'scan bad', 'bl bad'])
plt.title('driving_downtown 2s', fontsize=8)

plt.figure()
plt.subplot(4, 1, 1)
plt.bar(np.arange(0, 3), [0.14, 0.03, 0.0], width=0.1)
plt.bar(np.arange(3, 6), [0.03, 0.0, 0.0], width=0.1)
plt.xticks(np.arange(6), ['gt good', 'scan good', 'bl good',
                          'gt bad', 'scan bad', 'bl bad'])
plt.title('driving1 30s', fontsize=8)
plt.subplot(4, 1, 2)
plt.bar(np.arange(0, 3), [0.21, 0.16, 0], width=0.1)
plt.bar(np.arange(3, 6), [0.12, 0.09, 0.5], width=0.1)
plt.xticks(np.arange(6), ['gt good', 'scan good', 'bl good',
                          'gt bad', 'scan bad', 'bl bad'])
plt.title('driving1 10s', fontsize=8)

plt.subplot(4, 1, 3)
plt.bar(np.arange(0, 3), [0.28, 0.20, 0.0, ], width=0.1)
plt.bar(np.arange(3, 6), [0.16, 0.19, 0.5], width=0.1)
plt.xticks(np.arange(6), ['gt good', 'scan good', 'bl good',
                          'gt bad', 'scan bad', 'bl bad'])
plt.title('driving1 5s', fontsize=8)
plt.subplot(4, 1, 4)
plt.bar(np.arange(0, 3), [0.11, 0.25, 0.25, ], width=0.1)
plt.bar(np.arange(3, 6), [0.13, 0.34, 0.25], width=0.1)
plt.xticks(np.arange(6), ['gt good', 'scan good', 'bl good',
                          'gt bad', 'scan bad', 'bl bad'])
plt.title('driving1 2s', fontsize=8)
plt.show()
# plt.bar(np.arange(3), [0.173, 0.072, 0.33])
# 30s:
# driving_downtown:
#     original good percent = 0.17391304347826086, bad percent = 0.0
#     FaserRCNN scan good percent = 0.13043478260869565, bad percent = 0.0
#     Mobilenet scan good percent = 0.07246376811594203, bad percent = 0.0
#     Sampled(per 10) Mobilenet scan good percent = 0.07246376811594203, bad percent = 0.0
#     baseline good percent = 0.3333333333333333, bad percent = 0.0
#
#
# 10s driving_downtown:
#     original good percent = 0.27053140096618356, bad percent = 0.00966183574879227
#     FaserRCNN scan good percent = 0.21739130434782608, bad percent = 0.00966183574879227
#     Mobilenet scan good percent = 0.14009661835748793, bad percent = 0.00966183574879227
#     Sampled(per 50) Mobilenet scan good percent = 0.13043478260869565, bad percent = 0.01932367149758454
#     baseline good percent = 0.0, bad percent = 0.25
#
#
# 5s:
# driving_downtown:
#     original good percent = 0.3060240963855422, bad percent = 0.016867469879518072
#     FaserRCNN scan good percent = 0.26265060240963856, bad percent = 0.014457831325301205
#     Mobilenet scan good percent = 0.1686746987951807, bad percent = 0.021686746987951807
#     Sampled(per 50) Mobilenet scan good percent = 0.23373493975903614, bad percent = 0.021686746987951807
#     baseline good percent = 0.125, bad percent = 0.25
#
#
# 2s:
# driving_downtown:
#     original good percent = 0.15028901734104047, bad percent = 0.010597302504816955
#     FaserRCNN scan good percent = 0.2957610789980732, bad percent = 0.02023121387283237
#     Mobilenet scan good percent = 0.21290944123314065, bad percent = 0.03275529865125241
#     Sampled(per 50) Mobilenet scan good percent = 0.2996146435452794, bad percent = 0.046242774566473986
#     baseline good percent = 0.5714285714285714, bad percent = 0.0
