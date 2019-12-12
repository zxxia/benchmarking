"""Vigil Implementation."""
import numpy as np
import cv2


class Vigil():
    """Vigil Implementation."""

    def __init__(simple_model='mobilenent'):
        pass

    def crop_image(img, img_idx, simple_model_dets, tmp_folder):
        """Crop images based on simple model detections."""
        mask = np.zeros(img.shape, dtype=np.uint8)
        for box in simple_model_dets:
            xmin, ymin, xmax, ymax = box[:4]
            mask[ymin:ymax, xmin:xmax] = 1

        processed_img = img.copy()
        processed_img *= mask
        cv2.imwrite('{}/{:06d}.jpg'.format(tmp_folder, img_idx), processed_img)
        return mask
