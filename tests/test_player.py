import unittest
import sys
from unittest.mock import patch
from kabr_tools import player
from tests.utils import del_file


def run():
    player.main()


class TestPlayer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # TODO: download data
        pass

    @classmethod
    def tearDownClass(cls):
        # TODO: delete data
        pass

    def setUp(self):
        self.tool = "player.py"
        self.folder = "tests/behavior_example/DJI_0001"
        self.video = self.folder.rsplit("/", maxsplit=1)[-1]
        del_file(f"{self.folder}/{self.video}_demo.mp4")

    def tearDown(self):
        # TODO: delete outputs
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
