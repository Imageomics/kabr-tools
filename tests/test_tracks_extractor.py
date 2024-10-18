import unittest
import sys
from kabr_tools import tracks_extractor


class TestTracksExtractor(unittest.TestCase):
    def test_run(self):
        # run tracks_extractor
        sys.argv = ["tracks_extractor.py",
                    "--video", "tests/detection_example/DJI_0068.mp4",
                    "--annotation", "tests/detection_example/DJI_0068.xml"]
        tracks_extractor.main()

    def test_parse_arg_min(self):
        # parse arguments
        sys.argv = ["tracks_extractor.py",
                    "--video", "tests/detection_example/DJI_0068.mp4",
                    "--annotation", "tests/detection_example/DJI_0068.xml"]
        args = tracks_extractor.parse_args()
        self.assertEqual(args.video, "tests/detection_example/DJI_0068.mp4")
        self.assertEqual(
            args.annotation, "tests/detection_example/DJI_0068.xml")
        self.assertEqual(args.tracking, False)
        self.assertEqual(args.imshow, False)

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = ["tracks_extractor.py",
                    "--video", "tests/detection_example/DJI_0068.mp4",
                    "--annotation", "tests/detection_example/DJI_0068.xml",
                    "--tracking",
                    "--imshow"]
        args = tracks_extractor.parse_args()
        self.assertEqual(args.video, "tests/detection_example/DJI_0068.mp4")
        self.assertEqual(
            args.annotation, "tests/detection_example/DJI_0068.xml")
        self.assertEqual(args.tracking, True)
        self.assertEqual(args.imshow, True)
