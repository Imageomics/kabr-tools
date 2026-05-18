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

TESTSDIR = os.path.dirname(os.path.realpath(__file__))
REPOROOT = os.path.dirname(TESTSDIR)


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
        self.dataset = os.path.join(TESTSDIR, "ultralytics")
        self.skip = "1"
        self.label2index = os.path.join(REPOROOT, "ethogram", "label2index.json")

    def tearDown(self):
        # delete outputs
        del_dir(self.dataset)
        del_file(f"{self.dataset}.yaml")

    def test_run(self):
        # run cvat2ultralytics
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--annotation", self.annotation,
                    "--dataset", self.dataset,
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

        # check output — derive expected values from the annotation directly,
        # mirroring cvat2ultralytics.py logic so the test works for any input.
        from pathlib import Path as _Path
        annotations = etree.parse(TestCvat2Ultralytics.annotation).getroot()
        original_size = annotations.find("meta").find("task").find("original_size")
        height = int(original_size.find("height").text)
        width = int(original_size.find("width").text)

        track2end = {}
        for track in annotations.findall("track"):
            tid = int(track.attrib["id"])
            for box in track.findall("box"):
                if int(box.attrib["keyframe"]) == 1:
                    track2end[tid] = int(box.attrib["frame"])

        annotated = {}
        for track in annotations.findall("track"):
            tid = int(track.attrib["id"])
            for box in track.findall("box"):
                fid = int(box.attrib["frame"])
                if fid <= track2end[tid]:
                    annotated.setdefault(fid, []).append(box)

        skip = int(self.skip)
        processed = sorted(f for f in annotated if f % skip == 0)
        n = len(processed)
        val_start = int(n * 0.8)
        test_start = int(n * 0.87)
        name = _Path(TestCvat2Ultralytics.video).stem

        for i, frame_num in enumerate(processed):
            if i < val_start:
                folder = "train"
            elif i < test_start:
                folder = "val"
            else:
                folder = "test"

            im_path = f"{self.dataset}/images/{folder}/{name}_{frame_num}.jpg"
            label_path = f"{self.dataset}/labels/{folder}/{name}_{frame_num}.txt"
            self.assertTrue(file_exists(im_path))
            self.assertTrue(file_exists(label_path))

            img = cv2.imread(im_path)
            self.assertEqual(img.shape, (height, width, 3))

            data_label = pd.read_csv(label_path, sep=" ", header=None)
            expected_boxes = annotated[frame_num]
            self.assertEqual(len(data_label.index), len(expected_boxes))

            for j, box in enumerate(expected_boxes):
                x_c = (float(box.attrib["xtl"]) + float(box.attrib["xbr"])) / 2 / width
                y_c = (float(box.attrib["ytl"]) + float(box.attrib["ybr"])) / 2 / height
                w = (float(box.attrib["xbr"]) - float(box.attrib["xtl"])) / width
                h = (float(box.attrib["ybr"]) - float(box.attrib["ytl"])) / height
                row = data_label.iloc[j]
                self.assertEqual(row[0], 0)
                self.assertAlmostEqual(row[1], x_c, places=4)
                self.assertAlmostEqual(row[2], y_c, places=4)
                self.assertAlmostEqual(row[3], w, places=4)
                self.assertAlmostEqual(row[4], h, places=4)


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
