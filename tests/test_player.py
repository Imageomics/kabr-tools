import unittest
import sys
import os
from unittest.mock import patch
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
        self.video = self.folder.rsplit("/", maxsplit=1)[-1]

        # delete output
        del_file(f"{self.folder}/{self.video}_demo.mp4")

    def tearDown(self):
        # delete output
        del_file(f"{self.folder}/{self.video}_demo.mp4")

    @patch('kabr_tools.player.cv2.imshow')
    @patch('kabr_tools.player.cv2.namedWindow')
    @patch('kabr_tools.player.cv2.createTrackbar')
    @patch('kabr_tools.player.cv2.setTrackbarPos')
    @patch('kabr_tools.player.cv2.getTrackbarPos')
    def test_run(self, getTrackbarPos, setTrackbarPos, createTrackbar, namedWindow, imshow):
        # mock getTrackbarPos
        getTrackbarPos.return_value = 0

        # run player
        sys.argv = [self.tool,
                    "--folder", self.folder,
                    "--save"]
        run()
        self.assertTrue(file_exists(f"{self.folder}/{self.video}_demo.mp4"))

    @patch('kabr_tools.player.cv2.imshow')
    @patch('kabr_tools.player.cv2.namedWindow')
    @patch('kabr_tools.player.cv2.createTrackbar')
    @patch('kabr_tools.player.cv2.setTrackbarPos')
    @patch('kabr_tools.player.cv2.getTrackbarPos')
    def test_parse_arg_min(self, getTrackbarPos, setTrackbarPos, createTrackbar, namedWindow, imshow):
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
    @patch('kabr_tools.player.cv2.setTrackbarPos')
    @patch('kabr_tools.player.cv2.getTrackbarPos')
    def test_parse_arg_full(self, getTrackbarPos, setTrackbarPos, createTrackbar, namedWindow, imshow):
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
