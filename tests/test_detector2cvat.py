import unittest
import sys
from unittest.mock import patch
from kabr_tools import detector2cvat
from tests.utils import del_dir


def run():
    detector2cvat.main()

class TestDetector2Cvat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # TODO: download data
        pass

    @classmethod
    def tearDownClass(cls):
        # TODO: delete data
        pass

    def setUp(self):
        self.tool = "detector2cvat.py"
        self.video = "tests/detection_example"
        self.save = "tests/detection_example/output"

    def tearDown(self):
        # TODO: delete outputs
        del_dir(self.save)

    @patch('kabr_tools.detector2cvat.cv2.imshow')
    def test_run(self, imshow):
        # run detector2cvat
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", self.save]
        run()

    @patch('kabr_tools.detector2cvat.cv2.imshow')
    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", self.save]
        args = detector2cvat.parse_args()

        # check parsed argument values
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.save, self.save)

        # run detector2cvat
        run()
