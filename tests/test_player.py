import unittest
import sys
from unittest.mock import patch
from kabr_tools import player


class TestPlayer(unittest.TestCase):
    @patch('kabr_tools.player.cv2.imshow')
    @patch('kabr_tools.player.cv2.namedWindow')
    @patch('kabr_tools.player.cv2.createTrackbar')
    @patch('kabr_tools.player.cv2.setTrackbarPos')
    def test_run(self, imshow, namedWindow, createTrackbar, setTrackbarPos):
        # run player
        sys.argv = ["player.py",
                    "--folder", "tests/behavior_example/DJI_0001",
                    "--save"]
        player.main()
