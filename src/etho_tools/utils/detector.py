import numpy as np
import torch


class Detector:
    @staticmethod
    def get_centroid(detection):
        x = int(detection[0] + (detection[2] - detection[0]) / 2)
        y = int(detection[1] + (detection[3] - detection[1]) / 2)

        return x, y

    @staticmethod
    def get_crop(image, detection):
        return image[int(detection[1]):int(detection[3]), int(detection[0]):int(detection[2])]
