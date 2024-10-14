import unittest
import sys
from kabr_tools import cvat2ultralytics


class TestCvat2Ultralytics(unittest.TestCase):
    def test_cvat2ultralytics(self):
        sys.argv = ["cvat2ultralytics.py",
                    "--video", "tests/examples",
                    "--annotation", "tests/examples",
                    "--dataset", "tests/ultralytics"]
        cvat2ultralytics.main()
