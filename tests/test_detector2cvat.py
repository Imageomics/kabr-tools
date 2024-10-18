import unittest
import sys
from unittest.mock import patch
from kabr_tools import detector2cvat


class TestDetector2Cvat(unittest.TestCase):
    @patch('kabr_tools.miniscene2behavior.cv2.imshow')
    def test_detector(self, imshow):
        sys.argv = ["detector2cvat.py",
                    "--video", "tests/detection_example",
                    "--save", "tests/detection_example/output"]
        detector2cvat.main()
