import unittest
import sys
import os
from kabr_tools import cvat2slowfast
from tests.utils import (
    get_behavior,
    del_dir,
    del_file
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

        # run cvat2slowfast
        run()

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

        # run cvat2slowfast
        run()
