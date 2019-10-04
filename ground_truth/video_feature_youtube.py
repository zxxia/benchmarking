from utils.utils import load_metadata
from collections import defaultdict
from video_feature_KITTI import compute_para
from absl import app, flags
import pdb
from utils.model_utils import load_full_model_detection, filter_video_detections

FLAGS = flags.FLAGS
flags.DEFINE_string('resol', 'original', 'interested resolution.By default, '
                    'original resolution(the resolution of downloaded video)')
flags.DEFINE_string('input_file', None, 'Data path.')
flags.DEFINE_string('metadata_file', None, 'Metadta file.')
flags.DEFINE_string('output_file', None, 'Output feature file.')


# def read_annot(annot_path, target_object=[3, 8], score_range=[0.0, 1.0]):
#     all_filename = []
#     frame_to_object = defaultdict(list)
#     object_to_frame = defaultdict(list)
#     object_location = {}

#     with open(annot_path, 'r') as f:
#         f.readline()
#         for line in f:
#             # each line (frame_id, object_id, x, y, w, h, object_type, score)
#             line_list = line.strip().split(',')
#             frame_id = int(line_list[0])
#             try:
#                 object_id = int(line_list[1])
#             except ValueError:
#                 object_id = line_list[1]
#             if int(line_list[6]) not in target_object:
#                 continue
#             if len(line_list) == 8:
#                 score = float(line_list[7])
#                 if score < score_range[0] or score > score_range[1]:
#                     continue
#             frame_to_object[frame_id].append(line_list[1:])
#             object_to_frame[object_id].append(frame_id)
#             all_filename.append(frame_id)
#             key = (frame_id, object_id)
#             [x, y, w, h] = [int(float(x)) for x in line_list[2:6]]
#             object_location[key] = [x, y, w, h]
#     return all_filename, frame_to_object, object_to_frame, object_location


def main(argv):
    resol_dict = {'360p': [640, 360],
                  '480p': [854, 480],
                  '540p': [960, 540],
                  '576p': [1024, 576],
                  '720p': [1280, 720],
                  '1080p': [1920, 1080],
                  '2160p': [3840, 2160]}
    required_flags = ['metadata_file', 'input_file', 'output_file']
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))

    input_file = FLAGS.input_file
    output_file = FLAGS.output_file

    metadata = load_metadata(FLAGS.metadata_file)
    frame_rate = metadata['frame rate']
    if FLAGS.resol == 'original':
        image_resolution = metadata['resolution']
    else:
        image_resolution = resol_dict[FLAGS.resol]

    annot_path = input_file
    print(annot_path)

    # load the detetions and apply confidence score filter and
    detections, frame_cnt = load_full_model_detection(annot_path)
    detections = filter_video_detections(detections, target_types={3, 8})
    # import pdb
    # pdb.set_trace()

    paras = compute_para(detections, image_resolution, frame_rate)
    current_start = min(detections.keys())
    current_end = max(detections.keys())
    with open(output_file, 'w') as f:
        f.write('frame_id,num_of_object,object_area,arrival_rate,velocity,'
                'total_object_area,num_of_object_type,dominate_object_type\n')
        for frame_id in range(current_start, current_end + 1 - frame_rate):
            f.write(str(frame_id) + ',' +
                    str(paras.num_of_objects[frame_id]) + ',' +
                    str(paras.object_area[frame_id]) + ',' +
                    str(paras.arrival_rate[frame_id]) + ',' +
                    str(paras.velocity[frame_id]) + ',' +
                    str(paras.total_object_area[frame_id]) + ',' +
                    str(paras.object_type[frame_id]) + ',' +
                    str(paras.dominate_object_type[frame_id]) + '\n')


if __name__ == '__main__':
    app.run(main)
