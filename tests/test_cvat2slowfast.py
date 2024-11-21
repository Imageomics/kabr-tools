import unittest
import sys
from kabr_tools import cvat2slowfast
from tests.utils import (
    del_dir,
    dir_exists,
    file_exists
)

def run():
    cvat2slowfast.main()


class TestCvat2Slowfast(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # TODO: download data
        pass

    @classmethod
    def tearDownClass(cls):
        # TODO: delete data
        pass

    def setUp(self):
        self.tool = "cvat2slowfast.py"
        self.miniscene = "tests/behavior_example"
        self.dataset = "tests/slowfast"
        self.classes = "ethogram/classes.json"
        self.old2new = "ethogram/old2new.json"

    def tearDown(self):
        # TODO: delete outputs
        #del_dir(self.dataset)
        pass

    def test_run(self):
        # run cvat2slowfast
        sys.argv = [self.tool,
                    "--miniscene", self.miniscene,
                    "--dataset", self.dataset,
                    "--classes", self.classes]
        run()

        # check output dirs
        self.assertTrue(dir_exists(self.dataset))
        self.assertTrue(dir_exists(f"{self.dataset}/annotation"))
        self.assertTrue(dir_exists(f"{self.dataset}/dataset"))
        self.assertTrue(file_exists(f"{self.dataset}/annotation/classes.json"))
        self.assertTrue(file_exists(f"{self.dataset}/annotation/data.csv"))

        # check output files

        # TODO: check output dataset

    def test_parse_arg_min(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--miniscene", self.miniscene,
                    "--dataset", self.dataset,
                    "--classes", self.classes]
        args = cvat2slowfast.parse_args()

        # check parsed argument values
        self.assertEqual(args.miniscene, self.miniscene)
        self.assertEqual(args.dataset, self.dataset)
        self.assertEqual(args.classes, self.classes)

        # check default argument values
        self.assertEqual(args.old2new, None)

        # run cvat2slowfast
        run()

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = ["cvat2slowfast.py",
                    "--miniscene", self.miniscene,
                    "--dataset", self.dataset,
                    "--classes", self.classes,
                    "--old2new", self.old2new]
        args = cvat2slowfast.parse_args()

        # check parsed argument values
        self.assertEqual(args.miniscene, self.miniscene)
        self.assertEqual(args.dataset, self.dataset)
        self.assertEqual(args.classes, self.classes)
        self.assertEqual(args.old2new, self.old2new)

        # run cvat2slowfast
        run()
