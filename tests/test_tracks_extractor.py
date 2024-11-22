import unittest
import sys
from unittest.mock import patch
from kabr_tools import tracks_extractor
from tests.utils import del_dir


def run():
    tracks_extractor.main()


class TestTracksExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # TODO: download data
        pass

    @classmethod
    def tearDownClass(cls):
        # TODO: delete data
        pass

    def setUp(self):
        self.tool = "tracks_extractor.py"
        self.video = "tests/detection_example/DJI_0068.mp4"
        self.annotation = "tests/detection_example/DJI_0068.xml"

        # remove output directory before test
        del_dir("mini-scenes")

    def tearDown(self):
        # remove output directory after test
        del_dir("mini-scenes")

    def test_run(self):
        # run tracks_extractor
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--annotation", self.annotation]
        run()

    def test_parse_arg_min(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--annotation", self.annotation]
        args = tracks_extractor.parse_args()

        # check parsed arguments
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.annotation, self.annotation)

        # check default arguments
        self.assertEqual(args.tracking, False)
        self.assertEqual(args.imshow, False)

        # run tracks_extractor
        run()

    @patch('kabr_tools.tracks_extractor.cv2.imshow')
    def test_parse_arg_full(self, imshow):
        # parse arguments
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--annotation", self.annotation,
                    "--tracking",
                    "--imshow"]
        args = tracks_extractor.parse_args()

        # check parsed arguments
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.annotation, self.annotation)
        self.assertEqual(args.tracking, True)
        self.assertEqual(args.imshow, True)

        # run tracks_extractor
        run()
