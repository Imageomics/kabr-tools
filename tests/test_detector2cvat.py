import unittest
import sys
from unittest.mock import patch
from kabr_tools import detector2cvat


class TestDetector2Cvat(unittest.TestCase):
    def setUp(self):
        self.tool = "detector2cvat.py"
        self.video = "tests/detection_example"
        self.save = "tests/detection_example/output"

    @patch('kabr_tools.detector2cvat.cv2.imshow')
    def test_run(self, imshow):
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", self.save]
        detector2cvat.main()

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", self.save]
        args = detector2cvat.parse_args()

        # check parsed argument values
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.save, self.save)
