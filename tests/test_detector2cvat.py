import unittest
import sys
from kabr_tools import detector2cvat


class TestDetector2Cvat(unittest.TestCase):
    def test_detector(self):
        sys.argv = ["detector2cvat.py",
                    "--video", "detection_example/examples",
                    "--save", "detection_example/output"]
        detector2cvat.main()
