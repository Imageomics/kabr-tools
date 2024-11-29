import unittest
import sys
from unittest.mock import patch
from kabr_tools import tracks_extractor
from tests.utils import (
    get_cached_datafile,
    del_dir,
    del_file
)

VIDEO = "DJI_0068/DJI_0068.mp4"
ANNOTATION = "DJI_0068/DJI_0068.xml"

def run():
    tracks_extractor.main()


class TestTracksExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # download data
        cls.video = get_cached_datafile(VIDEO)
        cls.annotation = get_cached_datafile(ANNOTATION)

    @classmethod
    def tearDownClass(cls):
        # delete data
        del_file(cls.video)
        del_file(cls.annotation)

    def setUp(self):
        # set params
        self.tool = "tracks_extractor.py"
        self.video = TestTracksExtractor.video
        self.annotation = TestTracksExtractor.annotation

        # remove output directory
        del_dir("mini-scenes")

    def tearDown(self):
        # remove output directory
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
