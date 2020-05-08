"""Definition of Video class."""
import json
import os
import subprocess

import cv2


def read_metadata(video_filename):
    """Return the metadata of a video file(mp4)."""
    metadata = {}
    cmd = "ffprobe -v quiet -print_format json -show_format -show_streams "\
        "-select_streams v {}".format(video_filename).split()
    output = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
    output = json.loads(output)
    for stream in output['streams']:
        if stream['codec_type'] == 'video' and stream['codec_name'] == 'h264':
            metadata['resolution'] = (stream['width'], stream['height'])
            a, b = stream['avg_frame_rate'].split('/')
            metadata['frame rate'] = int(round(float(a) / float(b)))
            metadata['frame count'] = int(stream['nb_frames'])
            metadata['duration'] = float(stream['duration'])

    return metadata


class Video(object):
    """Class of Video."""

    def __init__(self, video_path, frames_folder=None):
        """Video Constructor."""
        # Sanity check
        if not os.path.exists(video_path) and not frames_folder:
            raise Exception(f'No {video_path}.in {video_path}.')

        self._frames_folder = frames_folder
        metadata = read_metadata(video_path)

        self._video_path = video_path
        # self._cache = cache
        self._frame_rate = metadata['frame rate']
        self._resolution = metadata['resolution']
        self._frame_count = metadata['frame count']
        # self._video_type = None
        self._quality_parameter = None  # quality_parameter

    @property
    def video_path(self):
        """Return video path."""
        return self._video_path

    @property
    def frame_rate(self):
        """Return video frame rate(fps)."""
        return self._frame_rate

    @property
    def resolution(self):
        """Return video resolution."""
        return self._resolution

    @property
    def frame_count(self):
        """Return number of frames in the video."""
        return self._frame_count

    @property
    def quality_parameter(self):
        """Return quality level of the video."""
        return self._quality_parameter

    # @property
    # def video_type(self):
    #     """Return video type."""
    #     # TODO: fix
    #     raise NotImplementedError
    #     # return self._video_type

    @property
    def duration(self):
        """Return video duration in seconds."""
        return self._frame_count / self._frame_rate

    def get_frame_image_name(self, frame_index):
        """Return the image file name at frame index."""
        assert 0 <= frame_index < self._frame_count, \
            f'{frame_index} is not in range [0, {self.count})'
        if not os.path.exits(self._frames_folder):
            os.mkdir(self._image_frames_folder)
        img_file = os.path.join(
            self._frames_folder, '{:06d}.jpg'.format(frame_index))
        return img_file

    def get_frame_image(self, frame_index):
        """Return the image at frame index."""
        img_path = self.get_frame_image_name(frame_index)
        assert 0 <= frame_index < self._frame_count, \
            f'{frame_index} is not in range [0, {self.count})'
        assert os.path.exists(
            img_path), f"{img_path} does not exist. Please decode first."
        img = cv2.imread(img_path)
        assert img.shape[0] == self._resolution[1] and \
            img.shape[1] == self._resolution[0]
        return img

    def get_frame_filesize(self, frame_index):
        """Return the image file size at frame index."""
        assert 0 <= frame_index < self._frame_count, \
            f'{frame_index} is not in range [0, {self.count})'
        filename = self.get_frame_image_name(frame_index)
        return os.path.getsize(filename)

    def encode(self, output_video_name, target_frame_indices=None,
               target_frame_rate=None, save_video=True, crf=25):
        """Encode the target frames into a video and return video size."""
        if os.path.exists(output_video_name):
            return os.path.getsize(output_video_name)
        print("start generating "+output_video_name)
        tmp_list_file = output_video_name + '_list.txt'
        with open(tmp_list_file, 'w') as f_list:
            for i in target_frame_indices:
                # based on sample rate, decide whether this frame is sampled
                line = 'file \'{}\'\n'.format(self.get_frame_image_name(i))
                f_list.write(line)

        # compress the sampled image into a video
        frame_size = '{}x{}'.format(self._resolution[0], self._resolution[1])

        cmd = 'ffmpeg -y -r {} -f concat -safe 0 -i {} -s {} -vcodec libx264 '\
            '-crf {} -pix_fmt yuv420p -hide_banner {}'.format(
                target_frame_rate, tmp_list_file, frame_size, crf,
                output_video_name)
        subprocess.run(cmd.split(' '), check=True)
        # get the video size
        video_size = os.path.getsize(output_video_name)
        if not save_video:
            os.remove(output_video_name)
        os.remove(tmp_list_file)
        print('target fps={}, target resolution={}, video size={}'
              .format(target_frame_rate, self._resolution, video_size))
        print('finish generating {} and size={}'.format(
            output_video_name, video_size))
        return video_size

    def encode_iframe_control(self, output_video_name,
                              target_frame_indices=None,
                              target_frame_rate=None, save_video=True):
        """Encode the target frames into a video and return video size.

        Encode I-frame every second.
        """
        if os.path.exists(output_video_name):
            return os.path.getsize(output_video_name)
        print("start generating "+output_video_name)
        tmp_list_file = output_video_name + '_list.txt'
        with open(tmp_list_file, 'w') as f_list:
            for i in target_frame_indices:
                # based on sample rate, decide whether this frame is sampled
                line = 'file \'{}\'\n'.format(self.get_frame_image_name(i))
                f_list.write(line)

        # compress the sampled image into a video
        frame_size = str(self._resolution[0]) + 'x' + str(self._resolution[1])

        cmd = 'ffmpeg -y -loglevel panic -r {} -f concat -safe 0 -i {} -s {} '\
            '-vcodec libx264 -crf {} -pix_fmt yuv420p -force_key_frames ' \
            'expr:gte(t,n_forced*1) -hide_banner {}'.format(
                target_frame_rate, tmp_list_file, frame_size,
                self._quality_level, output_video_name)
        subprocess.run(cmd.split(' '), check=True)
        # get the video size
        video_size = os.path.getsize(output_video_name)
        if not save_video:
            os.remove(output_video_name)
        os.remove(tmp_list_file)
        print('target fps={}, target resolution={}, video size={}'
              .format(target_frame_rate, self._resolution, video_size))
        print('finish generating {} and size={}'.format(
            output_video_name, video_size))
        return video_size

    # a better way to create a video
    def encode_ffmpeg(self, frame_range, every_n_frame, resolution,
                      quality_parameter, output_video_name=None):
        """Create a video using ffmepg."""
        # TODO: need to add frame rate tuning?
        # TODO: need to test the functionality
        assert isinstance(resolution, tuple) and len(resolution) == 2 and \
            all(isinstance(x, int) for x in resolution), "resolution must " \
            "be in then format of (int, int)"
        assert isinstance(quality_parameter, float) and \
            0 <= quality_parameter <= 51, 'quality_parameter must be a float '\
            'value in range [0, 51].'
        # output_video_name = os.path.join(output_path, "{}.mp4".format(name))
        cmd = "ffmpeg -y -i {} -an -vf " \
            "select=between(n\,{}\,{})*not(mod(n\,{})),setpts=PTS-STARTPTS " \
            "-vsync vfr -s {}x{} -crf {} -vcodec libx264 {} -hide_banner"\
            .format(self._video_path, frame_range[0], frame_range[1],
                    every_n_frame, resolution[0], resolution[1],
                    quality_parameter, output_video_name)
        print(cmd)
        subprocess.run(cmd.split(' '), check=True)

    def decode(self, resolution, frames_folder=None):
        """Extract frames from videos."""
        if frames_folder is not None:
            output_img_name = os.path.join(frames_folder, "%06d.jpg")
        else:
            if not os.path.exists(self._frames_folder):
                os.mkdir(self._output_path)
            output_img_name = os.path.join(self._output_path, "%06d.jpg")
        cmd = "ffmpeg -y -i {} -s {}x{} -start_number 0 -qscale:v 2 " \
            "-hide_banner {}".format(self._video_path, resolution[0],
                                     resolution[1], output_img_name)
        print(cmd)
        subprocess.run(cmd.split(' '), check=True)
