import unittest
import sys
import os
import shutil
from unittest.mock import patch
from kabr_tools import tracks_extractor


def del_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def run():
    tracks_extractor.main()


class TestTracksExtractor(unittest.TestCase):
    def setUp(self):
        self.tool = "tracks_extractor.py"
        self.video = "tests/detection_example/DJI_0068.mp4"
        self.annotation = "tests/detection_example/DJI_0068.xml"

        # Remove output directory before test
        del_dir("mini-scenes")

    def tearDown(self):
        # Remove output directory after test
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
        tracks_extractor.main()
