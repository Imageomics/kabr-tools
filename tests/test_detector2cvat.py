import unittest
import sys
import os
import shutil
from unittest.mock import MagicMock, patch
from kabr_tools import detector2cvat

SAVE_DIR = "tests/detector2cvat"

class TestDetector2Cvat(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        # Remove output files after all tests have been completed
        if os.path.exists(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)

    def setUp(self):
        self.tool = "detector2cvat.py"
        self.video = "tests/detection_example"

    @patch('kabr_tools.detector2cvat.cv2.imshow')
    def test_run(self, imshow):
        # Check if tool runs on real data
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", f"{SAVE_DIR}/run"]
        detector2cvat.main()


    @patch('kabr_tools.detector2cvat.cv2.imshow')
    @patch('kabr_tools.detector2cvat.YOLOv8')
    def test_mock_yolo(self, yolo, imshow):
        # Create fake YOLO
        yolo_instance = MagicMock()
        yolo_instance.forward.return_value = [[[0, 0, 0, 0], 0.95, 0]]
        yolo.get_centroid.return_value = (50, 50)
        yolo.return_value = yolo_instance

        # Run detector2cvat
        save = f"{SAVE_DIR}/mock/0"
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", save]
        detector2cvat.main()

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", SAVE_DIR]
        args = detector2cvat.parse_args()

        # check parsed argument values
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.save, SAVE_DIR)
