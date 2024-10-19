import unittest
import sys
from unittest.mock import patch
from kabr_tools import player


class TestPlayer(unittest.TestCase):
    @patch('kabr_tools.player.cv2.imshow')
    @patch('kabr_tools.player.cv2.namedWindow')
    @patch('kabr_tools.player.cv2.createTrackbar')
    @patch('kabr_tools.player.cv2.setTrackbarPos')
    @patch('kabr_tools.player.cv2.getTrackbarPos')
    def test_run(self, imshow, namedWindow, createTrackbar, setTrackbarPos, getTrackbarPos):
        # mock getTrackbarPos
        getTrackbarPos.return_value = 0

        # run player
        sys.argv = ["player.py",
                    "--folder", "tests/behavior_example/DJI_0001",
                    "--save"]
        player.main()

    def test_parse_arg_min(self):
        # parse arguments
        sys.argv = ["player.py",
                    "--folder", "tests/behavior_example/DJI_0001"]
        args = player.parse_args()
        self.assertEqual(args.folder, "tests/behavior_example/DJI_0001")
        self.assertEqual(args.save, False)

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = ["player.py",
                    "--folder", "tests/behavior_example/DJI_0001",
                    "--save"]
        args = player.parse_args()
        self.assertEqual(args.folder, "tests/behavior_example/DJI_0001")
        self.assertEqual(args.save, True)
