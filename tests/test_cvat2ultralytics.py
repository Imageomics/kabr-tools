import unittest
import sys
import os
from kabr_tools import cvat2ultralytics
from tests.utils import (
    del_dir,
    del_file,
    get_detection
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
        self.skip = "5"
        self.label2index = "ethogram/label2index.json"

    def tearDown(self):
        # delete outputs
        del_dir(self.dataset)

    def test_run(self):
        # run cvat2ultralytics
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--annotation", self.annotation,
                    "--dataset", "tests/ultralytics"]
        run()

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
        self.assertEqual(args.skip, 5)
        self.assertEqual(args.label2index, self.label2index)

        # run cvat2ultralytics
        run()
