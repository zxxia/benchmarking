from collections import defaultdict
import os
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import math
import numpy as np
import glob

from matplotlib.patches import Ellipse



def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def get_cov_ellipse(cov, centre, nstd):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(xy=centre, width=w, height=h, angle=theta)
    return ell


youtube_video_list = ['cropped_crossroad3','driving_downtown','highway','crossroad',
                    'crossroad2','crossroad3', 'crossroad4','crossroad5',
                    'driving1','driving2','crossroad6','crossroad7']
youtube_video_list = ['crossroad4']
# frame_rate_dict = {'walking': 30,
#                  'driving_downtown': 30, 
#                  'highway': 25, 
#                  'crossroad2': 30,
#                  'crossroad': 30,
#                  'crossroad3': 30,
#                  'crossroad4': 30,
#                  'crossroad5': 30,
#                  'driving1': 30,
#                  'driving2': 30,
#                  'crossroad6': 30,
#                  'crossroad7': 30}

def load_vs_performance(perf_file):
    f1_list = []
    fps_list = []
    with open(perf_file, 'r') as f:

        for line in f:
            print(line)
            _, _, f1, fps = line.strip().split(',')
            f1_list.append(float(f1))
            fps_list.append(float(fps))
            # index = line_list[0].split('_')[-1]
            # dataset_name = line_list[0].replace('_'+index, '')
            # if line_list[0] in target_videos:
            #     perf_dict[dataset_name].append((float(line_list[2]), float(line_list[3])))
    return f1_list, fps_list

def load_gl_performance(perf_file):
    f1_list = []
    fps_list = []
    with open(perf_file, 'r') as f:
        next(f) # skip the header
        for line in f:
            print(line)
            _, _, _, f1, fps = line.strip().split(',')
            f1_list.append(float(f1))
            fps_list.append(float(fps)) 
    #         index = line_list[0].split('_')[-1]
    #         dataset_name = line_list[0].replace('_'+index, '')
    #         f1 = float(line_list[2])
    #         if 'highway' in dataset_name:
    #             frame_rate = 25
    #         else:
    #             frame_rate = 30
    #         gpu_time = float(line_list[3])/frame_rate
    #         if f1 > 0:
    #             target_videos.append(line_list[0])
    #             perf_dict[dataset_name].append((f1, gpu_time))
    return f1_list, fps_list


def main():
    # dataset_name = []
    gl_perf_file = '/home/zxxia/benchmarking_orig/glimpse/glimpse_motivation_result_highway_no_traffic.csv'  
    f1_list_gl, gpu_time_list_gl = load_gl_performance(gl_perf_file)
    
    vs_perf_file = '/home/zxxia/benchmarking_orig/videostorm/videostorm_motivation_result_highway_no_traffic.csv'
    f1_list_vs, gpu_time_list_vs = load_vs_performance(vs_perf_file)
    
    nstd = 1
    # VS = load_VS_performance(VS_perf_file, target_videos)
    # # print(VS.skeys())
    # for key in youtube_video_list:
    #   fig, ax = plt.subplots(1,1, sharex=True)
    #   perf = VS[key]
    #   f1_list = []
    #   gpu_time_list = []
    #   for (f1, gpu_time) in perf:
    #       if f1 == 0:
    #           continue
    #       # ax.scatter(gpu_time, f1,color='royalblue')
    #       f1_list.append(f1)
    #       gpu_time_list.append(gpu_time)
    #   f1_mean = np.mean(f1_list)
    #   gpu_time_mean = np.mean(gpu_time_list)
    #   # ax.scatter(gpu_time_mean, f1_mean ,color='b')
    #   cov = np.cov(gpu_time_list, f1_list)
    #   e = get_cov_ellipse(cov, (gpu_time_mean, f1_mean), nstd)
    #   if e.height < 0.02:
    #       e.height = 0.02

    #   ax.add_artist(e)
    #   e.set_clip_box(ax.bbox)
    #   e.set_facecolor('b')
    #   f = open('ellipse_' + key + '.txt','w')
    #   f.write(' '.join([format(x,'5.3f') for x in [gpu_time_mean, f1_mean, e.width, e.height, e.angle]]) + '\n')
        

    #   perf = GL[key]
    #   f1_list = []
    #   gpu_time_list = []
    #   for (f1, gpu_time) in perf:
    #       if f1 == 0:
    #           continue
    #       # ax.scatter(gpu_time, f1,color='coral')
    #       f1_list.append(f1)
    #       gpu_time_list.append(gpu_time)
    

    ax = plt.gca()

    f1_mean_gl = np.mean(f1_list_gl)
    gpu_time_mean_gl = np.mean(gpu_time_list_gl)
        # ax.scatter(gpu_time_mean, f1_mean ,color='r')
    cov_gl = np.cov(gpu_time_list_gl, f1_list_gl)
    e_gl = get_cov_ellipse(cov_gl, (gpu_time_mean_gl, f1_mean_gl), nstd)
    #f.write(' '.join([format(x,'5.3f') for x in [gpu_time_mean, f1_mean, e.width, e.height, e.angle]])+ '\n')
    if e_gl.height < 0.02:
        e_gl.height = 0.02

    ax.add_artist(e_gl)
    e_gl.set_clip_box(ax.bbox)
    e_gl.set_facecolor('r')
    e_gl.set_alpha(0.5)
   


    f1_mean_vs = np.mean(f1_list_vs)
    gpu_time_mean_vs = np.mean(gpu_time_list_vs)
        # ax.scatter(gpu_time_mean, f1_mean ,color='r')
    cov_vs = np.cov(gpu_time_list_vs, f1_list_vs)
    e_vs = get_cov_ellipse(cov_vs, (gpu_time_mean_vs, f1_mean_vs), nstd)
    if e_vs.height < 0.02:
        e_vs.height = 0.02
    #f.write(' '.join([format(x,'5.3f') for x in [gpu_time_mean, f1_mean, e.width, e.height, e.angle]])+ '\n')
    
    ax.add_artist(e_vs)
    e_vs.set_clip_box(ax.bbox)
    e_vs.set_facecolor('b')
    e_vs.set_alpha(0.5)
    plt.legend([e_gl, e_vs], ['Glimpse', 'VideoStorm'], loc='lower right')
    plt.title("highway_no_traffic")
    plt.ylim([0,1.1])
    plt.xlim([0,1.1])
    plt.xlabel('GPU processing time')
    plt.ylabel('F1 score')
    plt.show()



    # para_path = '/Users/zhujunxiao/Desktop/benchmarking/Final_code/'\
    #           'dataset_parameter/paras/'
    # features = load_video_feature(para_path)
    # for i in range(0, 6):
    #   perf_vec = []
    #   feature_vec = []
    #   fig, ax = plt.subplots(1,1, sharex=True)
    #   for key in perf_dict.keys():
    #       if key not in features: 
    #           # print('feature not found:', key)
    #           continue
    #       if math.isnan(features[key][i]):
    #           continue
    #       perf_vec.append(perf_dict[key])
    #       feature_vec.append(features[key][i])
    #       ax.scatter(features[key][i], perf_dict[key],c='r')
    #   coef, p = spearmanr(perf_vec, feature_vec)
    #   alpha = 0.05
    #   if p > alpha:
    #       pass
    #   else:           
    #       print(feature_names[i])
    #       print('Spearmans correlation coefficient: %.3f' % coef)
    #       print('Samples are correlated (reject H0) p=%.3f' % p)                  
    #   plt.xlabel(feature_names[i])
    #   plt.ylabel('GPU processing time (ms)')
    #   plt.show()
    #   # print(features[key])


if __name__ == '__main__':
    main()
