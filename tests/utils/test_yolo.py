import unittest
from unittest.mock import MagicMock, patch
from collections import OrderedDict
import numpy as np
import torch
from ultralytics import YOLO
from kabr_tools.utils.yolo import YOLOv8

# from yolov8x.pt
LABELS = {"zebra": 22, "horse": 17, "giraffe": 23, "bear": 21}

def rescale(box, width, height):
    return [box[0] * width, box[1] * height, box[2] * width, box[3] * height]


class MockBox:
    def __init__(self, box=[[0, 0, 0, 0]], cls=["zebra"], conf=[0.95]):
        self.xyxyn = None
        self.cls = None
        self.conf = None

    def mock(self, boxes, classes, confs):
        self.xyxyn = torch.Tensor(boxes)
        self.cls = torch.Tensor([LABELS[cls] for cls in classes])
        self.conf = torch.Tensor(confs)
        return self


class TestYolo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.im = np.zeros((100, 101, 3), dtype=np.uint8)
        cls.box = OrderedDict([("x1", 10), ("y1", 20), ("x2", 30), ("y2", 40)])
        cls.box_values = list(cls.box.values())

    @patch("kabr_tools.utils.yolo.YOLO")
    def test_forward(self, yolo_mock):
        im = TestYolo.im
        yolo_model = MagicMock()
        yolo_model.predict.return_value.__getitem__ = lambda x, _: x
        yolo_model.names = YOLO("yolov8x.pt").names
        yolo_mock.return_value = yolo_model

        # horse -> zebra
        points = [[0] * 4] * 3
        labels = ["zebra", "horse", "giraffe"]
        expect_labels = ["Zebra", "Zebra", "Giraffe"]
        probs = [0.7, 0.8, 0.9]
        yolo_boxes = MockBox().mock(points, labels, probs)
        yolo_model.predict.return_value.boxes.cpu.return_value = yolo_boxes

        yolo = YOLOv8()
        preds = yolo.forward(im)

        self.assertEqual(len(preds), 3)
        for i, pred in enumerate(preds):
            self.assertEqual(preds[i][0], points[i])
            self.assertEqual(preds[i][1], probs[i])
            self.assertEqual(preds[i][2], expect_labels[i])

        # bear -> filtered
        points = [[0] * 4] * 3
        labels = ["bear", "horse", "giraffe"]
        expect_labels = [None, "Zebra", "Giraffe"]
        probs = [0.9, 0.8, 0.9]
        yolo_boxes = MockBox().mock(points, labels, probs)
        yolo_model.predict.return_value.boxes.cpu.return_value = yolo_boxes

        yolo = YOLOv8()
        preds = yolo.forward(im)

        self.assertEqual(len(preds), 2)
        index = 0
        for pred in preds:
            while expect_labels[index] is None:
                index += 1
            self.assertEqual(pred[0], rescale(points[index], im.shape[1], im.shape[0]))
            self.assertEqual(pred[1], probs[index])
            self.assertEqual(pred[2], expect_labels[index])
            index += 1
    
        # low prob -> filtered
        points = [[i] * 4 for i in range(8)]
        labels = ["bear", "horse", "zebra", "giraffe", "bear", "horse", "zebra", "giraffe"]
        expect_labels = [None, "Zebra", None, "Giraffe", None, "Zebra", None, None]
        probs = [0.5, 0.9, 0.4, 0.8, 0.7, 0.6, 0.3, 0.5]
        yolo_boxes = MockBox().mock(points, labels, probs)
        yolo_model.predict.return_value.boxes.cpu.return_value = yolo_boxes

        yolo = YOLOv8()
        preds = yolo.forward(im)

        self.assertEqual(len(preds), 3)
        index = 0
        for pred in preds:
            while expect_labels[index] is None:
                index += 1
            self.assertEqual(pred[0], rescale(points[index], im.shape[1], im.shape[0]))
            self.assertEqual(pred[1], probs[index])
            self.assertEqual(pred[2], expect_labels[index])
            index += 1

    def test_yolo_with_params(self):
        pass

    def test_get_centroid(self):
        box = TestYolo.box
        box_values = TestYolo.box_values
        x, y = YOLOv8.get_centroid(box_values)
        self.assertEqual(x, (box["x1"] + box["x2"]) // 2)
        self.assertEqual(y, (box["y1"] + box["y2"]) // 2)
