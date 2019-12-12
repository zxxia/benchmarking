""" Compute features """
from absl import app, flags
from utils.model_utils import load_full_model_detection, \
    filter_video_detections
from utils.utils import load_metadata
from video_feature_KITTI import compute_para
from constants import RESOL_DICT


FLAGS = flags.FLAGS
flags.DEFINE_string('resol', 'original', 'interested resolution.By default, '
                    'original resolution(the resolution of downloaded video)')
flags.DEFINE_string('input_file', None, 'Data path.')
flags.DEFINE_string('metadata_file', None, 'Metadta file.')
flags.DEFINE_string('output_file', None, 'Output feature file.')
flags.DEFINE_string('video_type', None, 'Video type')


def main(_):
    """ Compute video features """
    required_flags = ['metadata_file', 'input_file', 'output_file',
                      'video_type']
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))

    metadata = load_metadata(FLAGS.metadata_file)
    frame_rate = metadata['frame rate']
    if FLAGS.resol == 'original':
        resolution = metadata['resolution']
    else:
        resolution = RESOL_DICT[FLAGS.resol]

    # load the detetions and apply confidence score filter and
    detections, _ = load_full_model_detection(FLAGS.input_file)
    detections = filter_video_detections(detections, target_types={3, 8})
    if FLAGS.video_type == 'static':
        detections = filter_video_detections(detections,
                                             width_range=(0, resolution[0]/2),
                                             height_range=(resolution[0]//20,
                                                           resolution[1]/2),
                                             target_types={3, 8})
    else:
        detections = filter_video_detections(detections,
                                             height_range=(resolution[0]//20,
                                                           resolution[1]),
                                             target_types={3, 8})
    # road_trip
    # for frame_idx in detections:
    #     tmp_boxes = []
    #     for box in detections[frame_idx]:
    #         xmin, ymin, xmax, ymax = box[:4]
    #         if ymin >= 500 and ymax >= 500:
    #             continue
    #         if (xmax - xmin) >= 2/3 * 1280:
    #             continue
    #         tmp_boxes.append(box)
    #     detections[frame_idx] = tmp_boxes

    paras = compute_para(detections, resolution, frame_rate)
    current_start = min(detections.keys())
    current_end = max(detections.keys())
    with open(FLAGS.output_file, 'w') as f_out:
        f_out.write('frame_id,num_of_object,object_area,arrival_rate,velocity,'
                    'total_object_area,num_of_object_type,dominate_object_type'
                    '\n')
        for frame_id in range(current_start, current_end + 1 - frame_rate):
            f_out.write(str(frame_id) + ',' +
                        str(paras.num_of_objects[frame_id]) + ',' +
                        str(paras.object_area[frame_id]) + ',' +
                        str(paras.arrival_rate[frame_id]) + ',' +
                        str(paras.velocity[frame_id]) + ',' +
                        str(paras.total_object_area[frame_id]) + ',' +
                        str(paras.object_type[frame_id]) + ',' +
                        str(paras.dominate_object_type[frame_id]) + '\n')


if __name__ == '__main__':
    app.run(main)
