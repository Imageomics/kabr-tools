import unittest
import sys
from kabr_tools import cvat2ultralytics


class TestCvat2Ultralytics(unittest.TestCase):
    def setUp(self):
        self.tool = "cvat2ultralytics.py"
        self.video = "tests/detection_example"
        self.annotation = "tests/detection_example"
        self.dataset = "tests/ultralytics"
        self.skip = "5"
        self.label2index = "ethogram/label2index.json"

    def test_run(self):
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--annotation", self.annotation,
                    "--dataset", "tests/ultralytics"]
        cvat2ultralytics.main()

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
