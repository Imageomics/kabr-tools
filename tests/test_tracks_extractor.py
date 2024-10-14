import unittest
import sys
from kabr_tools import tracks_extractor


class TestTracksExtractor(unittest.TestCase):
    def test_extractor(self):
        # run tracks_extractor
        sys.argv = ["tracks_extractor.py",
                    "--video", "tests/examples/DJI_0068.mp4",
                    "--annotation", "tests/examples/tracks.xml"]
        tracks_extractor.main()
