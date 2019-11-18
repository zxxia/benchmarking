""" Some helper functions needed in feature scanning """
from collections import defaultdict
import numpy as np


def nonzero(orig_list):
    """ strip off all zero values in the input list """
    nonzero_list = [e for e in orig_list if e != 0]
    return nonzero_list


def nonan(orig_list):
    """ strip off all numpy nan in the input list """
    # nonzero_list = [e for e in orig_list if e != 0]
    nonan_list = [x for x in orig_list if not np.isnan(x)]
    return nonan_list


def parse_feature_line(line):
    """ parse a line in the raw feature file """
    assert line, "input line should not be empty"
    frame_id, obj_cnt, obj_area, arrival_rate, velocity, tot_obj_area, \
        obj_type, _ = line.strip().split(',')
    # , dominant_type
    if frame_id == '':
        frame_id = 0
    else:
        frame_id = int(frame_id)
    if obj_cnt == '':
        obj_cnt = 0
    else:
        obj_cnt = int(obj_cnt)
    if obj_area == '':
        obj_area = []
    else:
        obj_area = [float(x) for x in obj_area.split(' ')]

    if arrival_rate == '':
        arrival_rate = 0
    else:
        arrival_rate = int(arrival_rate)
    if velocity == '':
        velocity = []
    else:
        velocity = nonzero([float(x) for x in velocity.split(' ')])
    if tot_obj_area == '':
        tot_obj_area = 0
    else:
        tot_obj_area = float(tot_obj_area)
    if obj_type == '[]':
        obj_type = []
    else:
        obj_type = obj_type.split(' ')
    # if '3' not in features['object types'] or
    #  '8' not in features['object types']:
    #    print(features['object types'])
    # features['dominant type'] = dominant_type

    return frame_id, obj_cnt, obj_area, arrival_rate, velocity, tot_obj_area, \
        obj_type


def load_video_features(filename):
    """ Load video features into frame id mapping to frame features """
    frame_features = dict()
    with open(filename, 'r') as f_fts:
        f_fts.readline()
        for line in f_fts:
            frame_ft = {}
            frame_id, frame_ft['Object Count'], \
                frame_ft['Object Area'], frame_ft['Arrival Rate'], \
                frame_ft['Object Velocity'], \
                frame_ft['Total Object Area'], _ = parse_feature_line(line)
            frame_features[frame_id] = frame_ft
    return frame_features


def load_short_video_features(filename):
    """ Load video features computed from FasterRCNN detections """
    video_clips = []
    obj_cnt = []
    obj_area = []
    obj_velocity = []
    arriv_rate = []
    tot_area = []
    percent_frame_w_obj = []
    similarity = []
    with open(filename, 'r') as f_features:
        f_features.readline()
        for line in f_features:
            cols = line.strip().split(',')
            video_clips.append(cols[0])
            obj_cnt.append(float(cols[1]))
            obj_area.append(float(cols[3]))
            arriv_rate.append(float(cols[5]))
            obj_velocity.append(float(cols[7]))
            tot_area.append(float(cols[9]))
            percent_frame_w_obj.append(float(cols[11]))
            similarity.append(float(cols[12]))

    return video_clips, obj_cnt, obj_area, arriv_rate, obj_velocity, \
        tot_area, percent_frame_w_obj, similarity


# def load_selected_videos(video_list, perf_list, selected_videos):
#     ft_to_perf = defaultdict(list)
#     for video_name, perf in zip(video_list, perf_list):
#         for key in selected_videos.keys():
#             if video_name in selected_videos[key]:
#                 ft_to_perf[key].append(perf)
#     return ft_to_perf


def load_selected_data(videos, data, target_videos):
    """
    Load selected data from a list of data.
    videos: video names
    data: data should have the same length of videos
    target_videos: interested videos
    """
    assert len(videos) == len(data)
    selected_data = list()
    # videos = list()
    for video, val in zip(videos, data):
        if video in target_videos:
            selected_data.append(val)
            # videos.append(video_name)
    return selected_data


def load_glimpse_results(filename):
    """ Load glimplse result file """
    video_clips = []
    f1_score = []
    perf = []
    ideal_perf = []
    trigger_f1 = []
    with open(filename, 'r') as f_glimpse:
        f_glimpse.readline()
        for line in f_glimpse:
            cols = line.strip().split(',')
            video_clips.append(cols[0])
            f1_score.append(float(cols[3]))
            perf.append(float(cols[4]))
            if len(cols) == 6:
                ideal_perf.append(float(cols[5]))
            if len(cols) == 7:
                ideal_perf.append(float(cols[5]))
                trigger_f1.append(float(cols[6]))

    if len(cols) == 6:
        return video_clips, perf, f1_score, ideal_perf
    if len(cols) == 7:
        return video_clips, perf, f1_score, ideal_perf, trigger_f1
    return video_clips, perf, f1_score


def load_videostorm_results(filename):
    """ Load videostorm result file """
    videos = []
    perf_list = []
    acc_list = []
    with open(filename, 'r') as f_vs:
        f_vs.readline()
        for line in f_vs:
            line_list = line.strip().split(',')
            videos.append(line_list[0])
            perf_list.append(float(line_list[1]))

            if len(line_list) == 3:
                acc_list.append(float(line_list[2]))

    return videos, perf_list, acc_list


def load_videostorm_profile(filename):
    """ Load videostorm profiling file """
    videos = []
    perf_dict = defaultdict(list)
    acc_dict = defaultdict(list)
    with open(filename, 'r') as f_vs:
        f_vs.readline()  # remove headers
        for line in f_vs:
            line_list = line.strip().split(',')
            video = line_list[0]
            if video not in videos:
                videos.append(video)
            perf_dict[video].append(float(line_list[1]))
            # if len(line_list) == 3:
            acc_dict[video].append(float(line_list[2]))

    return videos, perf_dict, acc_dict


def load_awstream_profile(filename, size_filename):
    """ Load awstream profiling file """
    video_sizes = {}
    with open(size_filename, 'r') as f_size:
        for line in f_size:
            line_list = line.strip().split(',')
            video_sizes[line_list[0]] = float(line_list[1])
    videos = []
    resol_dict = defaultdict(list)
    acc_dict = defaultdict(list)
    size_dict = defaultdict(list)
    cnt_dict = defaultdict(list)
    with open(filename, 'r') as f_vs:
        f_vs.readline()  # remove headers
        for line in f_vs:
            line_list = line.strip().split(',')
            video = line_list[0]
            if video not in videos:
                videos.append(video)
            # print(int(line_list[1].strip('p')))
            resol_dict[video].append(int(line_list[1].strip('p')))
            # if len(line_list) == 3:
            acc_dict[video].append(float(line_list[3]))
            if video+'_'+line_list[1] in video_sizes:
                size_dict[video].append(video_sizes[video+'_'+line_list[1]])
            else:
                size_dict[video].append(0)
            cnt_dict[video].append(int(line_list[4])+int(line_list[6]))

    return videos, resol_dict, acc_dict, size_dict, cnt_dict


# def load_awstream_results(filename):
#     """ Load awstream result file """
#     # TODO: finish this function
#     videos = []
#     perf_list = []
#     acc_list = []
#     with open(filename, 'r') as f_vs:
#         f_vs.readline()
#         for line in f_vs:
#             line_list = line.strip().split(',')
#             videos.append(line_list[0])
#             perf_list.append(float(line_list[1]))
#
#             if len(line_list) == 3:
#                 acc_list.append(float(line_list[2]))
#
#     return videos, perf_list, acc_list


def load_30s_video_features(filename):
    """ Load video features computed from FasterRCNN detections """
    features = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(44))
    videos = np.loadtxt(filename, dtype=str, delimiter=',', skiprows=1,
                        usecols=(0))

    ret_features = []
    ret_videos = []
    for feature, video in zip(features, videos):
        if feature > 0:
            ret_features.append(feature)
            ret_videos.append(video)

    return ret_videos, ret_features


def num_frames_sampled(nb_frames, n_frame):
    sample_times = nb_frames // n_frame
    return sample_times * 2


def sample_video_features(video_features, metadata, short_video_length,
                          sample_rate):
    """ Sample video features """
    fps = metadata['frame rate']
    # resolution = metadata['resolution']
    frame_cnt = metadata['frame count']
    short_vid_features = defaultdict(lambda: {'Object Count': [],
                                              'Object Area': [],
                                              'Total Object Area': [],
                                              'Object Velocity': []})
    short_vid_to_frame_id = defaultdict(list)
    # sampled_velos = []
    print("vidoe frame count={}, features frame count={}"
          .format(frame_cnt, len(video_features.keys())))
    for fid in range(1, frame_cnt+1):
        short_vid = (fid-1)//(fps * short_video_length)
        if fid % sample_rate == 0 and fid in video_features:
            # print(frame_features[frame_idx])
            # sampled_velos.extend(video_features[fid]['Object Velocity'])
            short_vid_features[short_vid]['Object Velocity']\
                .extend(video_features[fid]['Object Velocity'])
            short_vid_features[short_vid]['Object Area'] \
                .extend(video_features[fid]['Object Area'])
            short_vid_features[short_vid]['Object Count'] \
                .append(video_features[fid]['Object Count'])
            short_vid_features[short_vid]['Total Object Area'] \
                .append(video_features[fid]['Total Object Area'])
            short_vid_to_frame_id[short_vid].append(fid)
    return short_vid_features, short_vid_to_frame_id
