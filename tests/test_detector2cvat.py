import unittest
import sys
from unittest.mock import MagicMock, patch
from kabr_tools import detector2cvat


class TestDetector2Cvat(unittest.TestCase):
    def setUp(self):
        self.tool = "detector2cvat.py"
        self.video = "tests/detection_example"
        self.save = "tests/detection_example/output"

    # @patch('kabr_tools.detector2cvat.cv2.imshow')
    # def test_run(self, imshow):
    #     sys.argv = [self.tool,
    #                 "--video", self.video,
    #                 "--save", self.save]
    #     detector2cvat.main()


    @patch('kabr_tools.detector2cvat.cv2.imshow')
    @patch('kabr_tools.detector2cvat.YOLOv8')
    def test_mock_yolo(self, yolo, imshow):

        # Create fake YOLO
        yolo_instance = MagicMock()
        yolo_instance.forward.return_value = [[[0, 0, 0, 0], 0.95, 0]]
        yolo.get_centroid.return_value = (50, 50)
        yolo.return_value = yolo_instance

        # Run detector2cvat
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
