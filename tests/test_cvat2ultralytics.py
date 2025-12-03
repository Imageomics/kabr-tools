import unittest
import sys
import os
from lxml import etree
import pandas as pd
import cv2
from kabr_tools import cvat2ultralytics
from tests.utils import (
    del_dir,
    del_file,
    get_detection,
    dir_exists,
    file_exists
)


def run():
    cvat2ultralytics.main()


class TestCvat2Ultralytics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # download data
        cls.video, cls.annotation = get_detection()
        cls.dir = os.path.dirname(os.path.dirname(cls.video))

    @classmethod
    def tearDownClass(cls):
        # delete data
        del_file(cls.video)
        del_file(cls.annotation)

    def setUp(self):
        self.tool = "cvat2ultralytics.py"
        self.video = TestCvat2Ultralytics.dir
        self.annotation = TestCvat2Ultralytics.dir
        self.dataset = "tests/ultralytics"
        self.skip = "1"
        self.label2index = "ethogram/label2index.json"

    def tearDown(self):
        # delete outputs
        del_dir(self.dataset)

    def test_run(self):
        # run cvat2ultralytics
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--annotation", self.annotation,
                    "--dataset", "tests/ultralytics",
                    "--skip", self.skip]
        run()

        # check for output dirs
        self.assertTrue(dir_exists(self.dataset))
        self.assertTrue(dir_exists(f"{self.dataset}/images/test"))
        self.assertTrue(dir_exists(f"{self.dataset}/images/train"))
        self.assertTrue(dir_exists(f"{self.dataset}/images/val"))
        self.assertTrue(dir_exists(f"{self.dataset}/labels/test"))
        self.assertTrue(dir_exists(f"{self.dataset}/labels/train"))
        self.assertTrue(dir_exists(f"{self.dataset}/labels/val"))

        # check output
        annotations = etree.parse(TestCvat2Ultralytics.annotation).getroot()
        tracks = [list(track.findall("box")) for track in annotations.findall("track")]
        self.assertEqual(len(tracks[0]), 21)
        self.assertEqual(len(tracks[0]), len(tracks[1]))
        original_size = annotations.find("meta").find("task").find("original_size")
        height = int(original_size.find("height").text)
        width = int(original_size.find("width").text)
        for i in range(len(tracks[0])):
            # check existence
            if i < 16:
                data_im = f"{self.dataset}/images/train/DJI_0068_{i}.jpg"
                self.assertTrue(file_exists(data_im))
                data_label = f"{self.dataset}/labels/train/DJI_0068_{i}.txt"
                self.assertTrue(file_exists(data_label))
            elif i < 18:
                data_im = f"{self.dataset}/images/val/DJI_0068_{i}.jpg"
                self.assertTrue(file_exists(data_im))
                data_label = f"{self.dataset}/labels/val/DJI_0068_{i}.txt"
                self.assertTrue(file_exists(data_label))
            else:
                data_im = f"{self.dataset}/images/test/DJI_0068_{i}.jpg"
                self.assertTrue(file_exists(data_im))
                data_label = f"{self.dataset}/labels/test/DJI_0068_{i}.txt"
                self.assertTrue(file_exists(data_label))

            # check image
            data_im = cv2.imread(data_im)
            self.assertEqual(data_im.shape, (height, width, 3))

            # check label
            data_label = pd.read_csv(data_label, sep = " ", header = None)
            annotation_label = []
            for track in tracks:
                box = track[i]
                x_start = float(box.attrib["xtl"])
                y_start = float(box.attrib["ytl"])
                x_end = float(box.attrib["xbr"])
                y_end = float(box.attrib["ybr"])
                x_center = (x_start + (x_end - x_start) / 2) / width
                y_center = (y_start + (y_end - y_start) / 2) / height
                w = (x_end - x_start) / width
                h = (y_end - y_start) / height
                annotation_label.append(
                    [0, x_center, y_center, w, h]
                )
            self.assertEqual(len(data_label.index), len(annotation_label))

            for i, label in enumerate(annotation_label):
                self.assertEqual(label[0], annotation_label[i][0])
                self.assertAlmostEqual(label[1], annotation_label[i][1], places=4)
                self.assertAlmostEqual(label[2], annotation_label[i][2], places=4)
                self.assertAlmostEqual(label[3], annotation_label[i][3], places=4)
                self.assertAlmostEqual(label[4], annotation_label[i][4], places=4)


    def test_parse_arg_min(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--annotation", self.annotation,
                    "--dataset", self.dataset]
        args = cvat2ultralytics.parse_args()

        # check parsed argument values
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.annotation, self.annotation)
        self.assertEqual(args.dataset, self.dataset)

        # check default argument values
        self.assertEqual(args.skip, 10)
        self.assertEqual(args.label2index, None)

        # run cvat2ultralytics
        run()

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--annotation", self.annotation,
                    "--dataset", self.dataset,
                    "--skip", self.skip,
                    "--label2index", self.label2index]
        args = cvat2ultralytics.parse_args()

        # check parsed argument values
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.annotation, self.annotation)
        self.assertEqual(args.dataset, self.dataset)
        self.assertEqual(args.skip, int(self.skip))
        self.assertEqual(args.label2index, self.label2index)

        # run cvat2ultralytics
        run()
