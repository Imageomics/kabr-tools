import unittest
import sys
from kabr_tools import tracks_extractor


class TestTracksExtractor(unittest.TestCase):
    def test_parse_args(self):
        sys.argv = ["tracks_extractor.py",
                    "--video", "tests/examples/DJI_0068.mp4",
                    "--annotation", "tests/examples/tracks.xml"]
        args = tracks_extractor.parse_args()
        tracks_extractor.tracks_extractor(args.video, args.annotation, args.tracking, args.imshow)
