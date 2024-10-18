import unittest
import sys
from kabr_tools import cvat2slowfast


class TestCvat2Slowfast(unittest.TestCase):
    def test_run(self):
        sys.argv = ["cvat2slowfast.py",
                    "--miniscene", "tests/behavior_example",
                    "--dataset", "tests/slowfast",
                    "--classes", "ethogram/classes.json"]
        cvat2slowfast.main()
