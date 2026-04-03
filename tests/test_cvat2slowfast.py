import unittest
import sys
import os
import json
import pandas as pd
import cv2
from kabr_tools import cvat2slowfast
from tests.test_tracks_extractor import (
    scene_width,
    scene_height
)
from tests.utils import (
    del_dir,
    del_file,
    dir_exists,
    file_exists,
    get_behavior
)


def run():
    cvat2slowfast.main()


class TestCvat2Slowfast(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # download data
        cls.video, cls.miniscene, cls.annotation, cls.metadata = get_behavior()
        cls.dir = os.path.dirname(os.path.dirname(cls.video))

    @classmethod
    def tearDownClass(cls):
        # delete data
        del_file(cls.video)
        del_file(cls.miniscene)
        del_file(cls.annotation)
        del_file(cls.metadata)
        del_dir(cls.dir)

    def setUp(self):
        # set params
        self.tool = "cvat2slowfast.py"
        self.miniscene = TestCvat2Slowfast.dir
        self.dataset = "tests/slowfast"
        self.classes = "ethogram/classes.json"
        self.old2new = "ethogram/old2new.json"

    def tearDown(self):
        # delete outputs
        del_dir(self.dataset)

    def test_run(self):
        # run cvat2slowfast
        sys.argv = [self.tool,
                    "--miniscene", self.miniscene,
                    "--dataset", self.dataset,
                    "--classes", self.classes]
        run()

        # check output dirs
        self.assertTrue(dir_exists(self.dataset))
        self.assertTrue(dir_exists(f"{self.dataset}/annotation"))
        self.assertTrue(dir_exists(f"{self.dataset}/dataset/image"))
        self.assertTrue(file_exists(f"{self.dataset}/annotation/classes.json"))
        self.assertTrue(file_exists(f"{self.dataset}/annotation/data.csv"))

        # check classes.json
        with open(f"{self.dataset}/annotation/classes.json", "r", encoding="utf-8") as f:
            classes = json.load(f)
        with open(self.classes, "r", encoding="utf-8") as f:
            ethogram = json.load(f)
        self.assertEqual(classes, ethogram)

        # check data.csv
        with open(f"{self.dataset}/annotation/data.csv", "r", encoding="utf-8") as f:
            df = pd.read_csv(f, sep=" ")

        video_id = 1
        for i, row in df.iterrows():
            self.assertEqual(row["original_vido_id"], f"Z{video_id:04d}")
            self.assertEqual(row["video_id"], video_id)
            self.assertEqual(row["frame_id"], i+1)
            self.assertEqual(row["path"], f"Z{video_id:04d}/{i+1}.jpg")
            self.assertEqual(row["labels"], 1)
        self.assertEqual(i, 90)

        # check dataset
        for i in range(1, 92):
            data_im = f"{self.dataset}/dataset/image/Z{video_id:04d}/{i}.jpg"
            self.assertTrue(file_exists(data_im))
            data_im = cv2.imread(data_im)
            self.assertEqual(data_im.shape, (scene_height, scene_width, 3))

    def test_parse_arg_min(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--miniscene", self.miniscene,
                    "--dataset", self.dataset,
                    "--classes", self.classes]
        args = cvat2slowfast.parse_args()

        # check parsed argument values
        self.assertEqual(args.miniscene, self.miniscene)
        self.assertEqual(args.dataset, self.dataset)
        self.assertEqual(args.classes, self.classes)

        # check default argument values
        self.assertEqual(args.old2new, None)
        self.assertTrue(not args.no_images)

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = ["cvat2slowfast.py",
                    "--miniscene", self.miniscene,
                    "--dataset", self.dataset,
                    "--classes", self.classes,
                    "--old2new", self.old2new,
                    "--no_images"]
        args = cvat2slowfast.parse_args()

        # check parsed argument values
        self.assertEqual(args.miniscene, self.miniscene)
        self.assertEqual(args.dataset, self.dataset)
        self.assertEqual(args.classes, self.classes)
        self.assertEqual(args.old2new, self.old2new)
        self.assertTrue(args.no_images)
