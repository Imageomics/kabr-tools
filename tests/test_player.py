import unittest
import sys
import os
from pathlib import Path
import cv2 as real_cv2
from unittest.mock import patch, MagicMock
from kabr_tools import player
from tests.utils import (
    del_file,
    del_dir,
    file_exists,
    get_behavior
)


def run():
    player.main()


class TestPlayer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # download data
        cls.video, cls.miniscene, cls.annotation, cls.metadata = get_behavior()
        cls.dir = os.path.dirname(cls.video)

    @classmethod
    def tearDownClass(cls):
        # delete data
        del_file(cls.video)
        del_file(cls.miniscene)
        del_file(cls.annotation)
        del_file(cls.metadata)
        del_dir(cls.dir)

    def setUp(self):
        # set params
        self.tool = "player.py"
        self.folder = TestPlayer.dir
        self.video = Path(self.folder).name

        # delete output
        del_file(f"{self.folder}/{self.video}_demo.mp4")

    def tearDown(self):
        # delete output
        del_file(f"{self.folder}/{self.video}_demo.mp4")

    @patch('kabr_tools.player.cv2.imshow')
    @patch('kabr_tools.player.cv2.namedWindow')
    @patch('kabr_tools.player.cv2.createTrackbar')
    @patch('kabr_tools.player.cv2.setTrackbarMax')
    @patch('kabr_tools.player.cv2.setTrackbarPos')
    @patch('kabr_tools.player.cv2.getTrackbarPos')
    def test_run(self, getTrackbarPos, setTrackbarPos, setTrackbarMax, createTrackbar, namedWindow, imshow):
        # mock getTrackbarPos
        getTrackbarPos.return_value = 0

        # run player
        sys.argv = [self.tool,
                    "--folder", self.folder,
                    "--save"]
        run()
        self.assertTrue(file_exists(f"{self.folder}/{self.video}_demo.mp4"))

        cap = real_cv2.VideoCapture(f"{self.folder}/{self.video}.mp4")
        expected_max = int(cap.get(real_cv2.CAP_PROP_FRAME_COUNT)) - 1
        cap.release()
        createTrackbar.assert_called_once_with(
            self.video, "TrackPlayer", 0, expected_max, player.on_slider_change
        )

    @patch('kabr_tools.player.cv2.imshow')
    @patch('kabr_tools.player.cv2.namedWindow')
    @patch('kabr_tools.player.cv2.createTrackbar')
    @patch('kabr_tools.player.cv2.setTrackbarMax')
    @patch('kabr_tools.player.cv2.setTrackbarPos')
    @patch('kabr_tools.player.cv2.getTrackbarPos')
    def test_parse_arg_min(self, getTrackbarPos, setTrackbarPos, setTrackbarMax, createTrackbar, namedWindow, imshow):
        # parse arguments
        sys.argv = [self.tool,
                    "--folder", self.folder]
        args = player.parse_args()

        # check parsed arguments
        self.assertEqual(args.folder, self.folder)

        # check default arguments
        self.assertEqual(args.save, False)

        # run player
        run()
        self.assertTrue(not file_exists(f"{self.folder}/{self.video}_demo.mp4"))

    @patch('kabr_tools.player.cv2.imshow')
    @patch('kabr_tools.player.cv2.namedWindow')
    @patch('kabr_tools.player.cv2.createTrackbar')
    @patch('kabr_tools.player.cv2.setTrackbarMax')
    @patch('kabr_tools.player.cv2.setTrackbarPos')
    @patch('kabr_tools.player.cv2.getTrackbarPos')
    def test_parse_arg_full(self, getTrackbarPos, setTrackbarPos, setTrackbarMax, createTrackbar, namedWindow, imshow):
        # parse arguments
        sys.argv = [self.tool,
                    "--folder", self.folder,
                    "--save", "--imshow"]
        args = player.parse_args()

        # check parsed arguments
        self.assertEqual(args.folder, self.folder)
        self.assertEqual(args.save, True)

        # run player
        run()
        self.assertTrue(file_exists(f"{self.folder}/{self.video}_demo.mp4"))

    @patch('kabr_tools.player.cv2.setTrackbarMax')
    @patch('kabr_tools.player.cv2.setTrackbarPos')
    def test_update_trackbar(self, setTrackbarPos, setTrackbarMax):
        mock_cap = MagicMock()
        mock_cap.get.return_value = 50.0
        player.vcs = {"43": mock_cap}
        player.name = self.video
        player.index = 10

        player.update_trackbar("43")

        setTrackbarMax.assert_called_once_with(self.video, "TrackPlayer", 49)
        setTrackbarPos.assert_not_called()
