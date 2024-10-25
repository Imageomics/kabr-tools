import unittest
import sys
from kabr_tools import cvat2slowfast


class TestCvat2Slowfast(unittest.TestCase):
    def setUp(self):
        self.tool = "cvat2slowfast.py"
        self.miniscene = "tests/behavior_example"
        self.dataset = "tests/slowfast"
        self.classes = "ethogram/classes.json"
        self.old2new = "ethogram/old2new.json"

    def test_run(self):
        sys.argv = [self.tool,
                    "--miniscene", self.miniscene,
                    "--dataset", self.dataset,
                    "--classes", self.classes]
        cvat2slowfast.main()

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

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = ["cvat2slowfast.py",
                    "--miniscene", self.miniscene,
                    "--dataset", self.dataset,
                    "--classes", self.classes,
                    "--old2new", self.old2new]
        args = cvat2slowfast.parse_args()

        # check parsed argument values
        self.assertEqual(args.miniscene, self.miniscene)
        self.assertEqual(args.dataset, self.dataset)
        self.assertEqual(args.classes, self.classes)
        self.assertEqual(args.old2new, self.old2new)
