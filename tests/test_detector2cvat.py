import unittest
import sys
from unittest.mock import patch
from kabr_tools import detector2cvat


class TestDetector2Cvat(unittest.TestCase):
    @patch('kabr_tools.detector2cvat.cv2.imshow')
    def test_run(self, imshow):
        sys.argv = ["detector2cvat.py",
                    "--video", "tests/detection_example",
                    "--save", "tests/detection_example/output"]
        detector2cvat.main()

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = ["detector2cvat.py",
                    "--video", "tests/detection_example",
                    "--save", "tests/detection_example/output"]
        args = detector2cvat.parse_args()
        self.assertEqual(args.video, "tests/detection_example")
        self.assertEqual(args.save, "tests/detection_example/output")
