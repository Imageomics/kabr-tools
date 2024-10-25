import unittest
import sys
from unittest.mock import patch
from kabr_tools import player


class TestPlayer(unittest.TestCase):
    def setUp(self):
        self.tool = "player.py"
        self.folder = "tests/behavior_example/DJI_0001"

    @patch('kabr_tools.player.cv2.imshow')
    @patch('kabr_tools.player.cv2.namedWindow')
    @patch('kabr_tools.player.cv2.createTrackbar')
    @patch('kabr_tools.player.cv2.setTrackbarPos')
    @patch('kabr_tools.player.cv2.getTrackbarPos')
    def test_run(self, imshow, namedWindow, createTrackbar, setTrackbarPos, getTrackbarPos):
        # mock getTrackbarPos
        getTrackbarPos.return_value = 0

        # run player
        sys.argv = [self.tool,
                    "--folder", self.folder,
                    "--save"]
        player.main()

    def test_parse_arg_min(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--folder", self.folder]
        args = player.parse_args()

        # check parsed arguments
        self.assertEqual(args.folder, self.folder)

        # check default arguments
        self.assertEqual(args.save, False)

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--folder", self.folder,
                    "--save"]
        args = player.parse_args()

        # check parsed arguments
        self.assertEqual(args.folder, self.folder)
        self.assertEqual(args.save, True)
