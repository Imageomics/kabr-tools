import unittest
import sys
from kabr_tools import cvat2ultralytics


class TestCvat2Ultralytics(unittest.TestCase):
    def test_run(self):
        sys.argv = ["cvat2ultralytics.py",
                    "--video", "tests/detection_example",
                    "--annotation", "tests/detection_example",
                    "--dataset", "tests/ultralytics"]
        cvat2ultralytics.main()

    def test_parse_arg_min(self):
        # parse arguments
        sys.argv = ["cvat2ultralytics.py",
                    "--video", "tests/detection_example",
                    "--annotation", "tests/detection_example",
                    "--dataset", "tests/ultralytics"]
        args = cvat2ultralytics.parse_args()
        self.assertEqual(args.video, "tests/detection_example")
        self.assertEqual(args.annotation, "tests/detection_example")
        self.assertEqual(args.dataset, "tests/ultralytics")
        self.assertEqual(args.skip, 10)
        self.assertEqual(args.label2index, None)

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = ["cvat2ultralytics.py",
                    "--video", "tests/detection_example",
                    "--annotation", "tests/detection_example",
                    "--dataset", "tests/ultralytics",
                    "--skip", "5",
                    "--label2index", "ethogram/label2index.json"]
        args = cvat2ultralytics.parse_args()
        self.assertEqual(args.video, "tests/detection_example")
        self.assertEqual(args.annotation, "tests/detection_example")
        self.assertEqual(args.dataset, "tests/ultralytics")
        self.assertEqual(args.skip, 5)
        self.assertEqual(args.label2index, "ethogram/label2index.json")
