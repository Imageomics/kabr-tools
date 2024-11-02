import unittest
import sys
import os
import shutil
from lxml import etree
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
        yolo_instance.forward.return_value = [[[0, 0, 0, 0], 0.95, 'Grevy']]
        yolo.get_centroid.return_value = (50, 50)
        yolo.return_value = yolo_instance

        # Run detector2cvat
        save = f"{SAVE_DIR}/mock/0"
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", save]
        detector2cvat.main()

        # Check output exists
        output_path = f"{save}/{self.video}/DJI_0068.xml"
        self.assertTrue(os.path.exists(output_path))

        # Check output content
        xml_content = etree.parse(output_path).getroot()
        self.assertEqual(xml_content.tag, "annotations")
        for i, track in enumerate(xml_content.findall("track")):
            track_len = len(track.findall("box"))
            self.assertEqual(track.get("id"), str(i+1))
            self.assertEqual(track.get("label"), "Grevy")
            # TODO: Check if source should be manual
            self.assertEqual(track.get("source"), "manual")
            for i, box in enumerate(track.findall("box")):
                self.assertEqual(box.get("frame"), str(i))
                self.assertEqual(box.get("xtl"), "0.00")
                self.assertEqual(box.get("ytl"), "0.00")
                self.assertEqual(box.get("xbr"), "0.00")
                self.assertEqual(box.get("ybr"), "0.00")
                self.assertEqual(box.get("occluded"), "0")
                self.assertEqual(box.get("keyframe"), "1")
                self.assertEqual(box.get("z_order"), "0")
                # tracker marks last box as outside
                if i == track_len - 1:
                    self.assertEqual(box.get("outside"), "1")
                else:
                    self.assertEqual(box.get("outside"), "0")

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", SAVE_DIR]
        args = detector2cvat.parse_args()

        # check parsed argument values
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.save, SAVE_DIR)
