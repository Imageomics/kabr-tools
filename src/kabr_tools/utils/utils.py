import numpy as np
import cv2


def get_scene(image, object, scene_width, scene_height):
    width = scene_width // 2
    height = scene_height // 2
    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0
    size = object.box[3] - object.box[1]

    if object.box is not None and size > scene_height:
        coefficient = size / scene_height
        margin = 50
        start_x = np.max([0, object.centroid[0] - int(width * coefficient) - margin])
        start_y = np.max([0, object.box[1] - margin])
        end_x = np.min([image.shape[1] - 1, object.centroid[0] + int(width * coefficient) + margin])
        end_y = np.min([image.shape[0] - 1, object.box[3] + margin])
    else:
        start_x = np.max([0, object.centroid[0] - width])
        start_y = np.max([0, object.centroid[1] - height])
        end_x = np.min([image.shape[1] - 1, object.centroid[0] + width])
        end_y = np.min([image.shape[0] - 1, object.centroid[1] + height])

    crop = image[start_y:end_y, start_x:end_x]

    if object.centroid[0] - width < 0:
        pad_left = width - object.centroid[0]

    if object.centroid[1] - height < 0:
        pad_top = height - object.centroid[1]

    if object.centroid[0] + width > image.shape[1] - 1:
        pad_right = object.centroid[0] + width - image.shape[1] + 1

    if object.centroid[1] + height > image.shape[0] - 1:
        pad_bottom = object.centroid[1] + height - image.shape[0] + 1

    padded = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return cv2.resize(padded, (scene_width, scene_height), interpolation=cv2.INTER_AREA)
