import unittest
import sys
import os
from kabr_tools import detector2cvat
from tests.utils import (
    del_dir,
    del_file,
    get_detection
)


def run():
    detector2cvat.main()

class TestDetector2Cvat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # download data
        cls.video, cls.annotation = get_detection()
        cls.dir = os.path.dirname(cls.video)

    @classmethod
    def tearDownClass(cls):
        # delete data
        del_file(cls.video)
        del_file(cls.annotation)
        del_dir(cls.dir)

    def setUp(self):
        # set params
        self.tool = "detector2cvat.py"
        self.video = TestDetector2Cvat.dir
        self.save = "tests/detector2cvat"

    def tearDown(self):
        # delete outputs
        del_dir(self.save)

    def test_run(self):
        # run detector2cvat
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", self.save]
        run()

    def test_parse_arg_min(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", self.save]
        args = detector2cvat.parse_args()

        # check parsed argument values
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.save, self.save)
        self.assertEqual(args.imshow, False)

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", self.save,
                    "--imshow"]
        args = detector2cvat.parse_args()

        # check parsed argument values
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.save, self.save)
        self.assertEqual(args.imshow, True)
