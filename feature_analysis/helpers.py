""" Some helper functions needed in feature scanning """


def nonzero(orig_list):
    """ strip off all zero values in the input list """
    nonzero_list = [e for e in orig_list if e != 0]
    return nonzero_list


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
