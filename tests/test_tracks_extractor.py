import unittest
import sys
from kabr_tools import tracks_extractor


class TestTracksExtractor(unittest.TestCase):
    def setUp(self):
        self.tool = "tracks_extractor.py"
        self.video = "tests/behavior_example/DJI_0001"
        self.annotation = "tests/behavior_example/DJI_0001"


    def test_run(self):
        # run tracks_extractor
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--annotation", self.annotation]
        tracks_extractor.main()

    def test_parse_arg_min(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--annotation", self.annotation]
        args = tracks_extractor.parse_args()

        # check parsed arguments
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.annotation,self.annotation)

        # check default arguments
        self.assertEqual(args.tracking, False)
        self.assertEqual(args.imshow, False)

    def test_parse_arg_full(self):
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
