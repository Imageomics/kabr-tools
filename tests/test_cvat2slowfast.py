import unittest
import sys
from kabr_tools import cvat2slowfast


class TestCvat2Slowfast(unittest.TestCase):
    def test_run(self):
        sys.argv = ["cvat2slowfast.py",
                    "--miniscene", "tests/behavior_example",
                    "--dataset", "tests/slowfast",
                    "--classes", "ethogram/classes.json"]
        cvat2slowfast.main()

    def test_parse_arg_min(self):
        # parse arguments
        sys.argv = ["cvat2slowfast.py",
                    "--miniscene", "tests/behavior_example",
                    "--dataset", "tests/slowfast",
                    "--classes", "ethogram/classes.json"]
        args = cvat2slowfast.parse_args()
        self.assertEqual(args.miniscene, "tests/behavior_example")
        self.assertEqual(args.dataset, "tests/slowfast")
        self.assertEqual(args.classes, "ethogram/classes.json")
        self.assertEqual(args.old2new, None)

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = ["cvat2slowfast.py",
                    "--miniscene", "tests/behavior_example",
                    "--dataset", "tests/slowfast",
                    "--classes", "ethogram/classes.json",
                    "--old2new", "ethogram/old2new.json"]
        args = cvat2slowfast.parse_args()
        self.assertEqual(args.miniscene, "tests/behavior_example")
        self.assertEqual(args.dataset, "tests/slowfast")
        self.assertEqual(args.classes, "ethogram/classes.json")
        self.assertEqual(args.old2new, "ethogram/old2new.json")
