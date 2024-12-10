import unittest
import sys
import os
from lxml import etree
from unittest.mock import MagicMock, patch
import cv2
from lxml import etree
import numpy as np
from kabr_tools import detector2cvat
from kabr_tools.utils.yolo import YOLOv8
from tests.utils import (
    del_dir,
    del_file,
    file_exists,
    get_detection
)


class DetectionData:
    def __init__(self, video_dim, video_len, annotation):
        self.video_dim = video_dim
        self.video_len = video_len
        self.frame = -1
        self.video_frame = np.zeros(video_dim, dtype=np.uint8)

        annotation = etree.parse(annotation).getroot()
        self.tracks = [[0, list(track.findall("box")),
                        track.get("label")]
                       for track in annotation.findall("track")]

    def read(self):
        if self.frame >= self.video_len - 1:
            return False, None
        self.frame += 1
        return True, self.video_frame

    def get(self, param):
        if param == cv2.CAP_PROP_FRAME_COUNT:
            return self.video_len
        elif param == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.video_dim[0]
        elif param == cv2.CAP_PROP_FRAME_WIDTH:
            return self.video_dim[1]
        else:
            return None

    def forward(self, data):
        soln = []
        for track in self.tracks:
            ptr = track[0]
            if ptr < len(track[1]):
                box = track[1][ptr]
                frame = int(box.get("frame"))
                if frame == self.frame:
                    soln.append([[int(float(box.get("xtl"))),
                                int(float(box.get("ytl"))),
                                int(float(box.get("xbr"))),
                                int(float(box.get("ybr")))],
                                0.95,
                                track[2]])
                    track[0] += 1
        return soln


def run():
    detector2cvat.main()


class TestDetector2Cvat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # download data
        cls.video, cls.annotation = get_detection()
        cls.dir = os.path.dirname(cls.video)

    @classmethod
    def tearDownClass(cls):
        # delete data
        del_file(cls.video)
        del_file(cls.annotation)
        del_dir(cls.dir)

    def setUp(self):
        # set params
        self.tool = "detector2cvat.py"
        self.video = TestDetector2Cvat.dir
        self.save = "tests/detector2cvat"
        self.dir = "/".join(os.path.splitext(self.video)[0].split('/')[-2:])

    def tearDown(self):
        # delete outputs
        del_dir(self.save)

    def test_run(self):
        # check if tool runs on real data
        save = f"{self.save}/run"
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", save]
        detector2cvat.main()

        # check output exists
        output_path = f"{save}/{self.dir}/DJI_0068.xml"
        self.assertTrue(file_exists(output_path))
        demo_path = f"{save}/{self.dir}/DJI_0068_demo.mp4"
        self.assertTrue(file_exists(demo_path))

    @patch('kabr_tools.detector2cvat.YOLOv8')
    def test_mock_yolo(self, yolo):
        # create fake YOLO
        yolo_instance = MagicMock()
        yolo_instance.forward.return_value = [[[0, 0, 0, 0], 0.95, 'Grevy']]
        yolo.get_centroid.return_value = (50, 50)
        yolo.return_value = yolo_instance

        # run detector2cvat
        save = f"{self.save}/mock/0"
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", save]
        detector2cvat.main()

        # check output exists
        output_path = f"{save}/{self.dir}/DJI_0068.xml"
        self.assertTrue(file_exists(output_path))
        demo_path = f"{save}/{self.dir}/DJI_0068_demo.mp4"
        self.assertTrue(file_exists(demo_path))

        # check output xml
        xml_content = etree.parse(output_path).getroot()
        self.assertEqual(xml_content.tag, "annotations")
        for j, track in enumerate(xml_content.findall("track")):
            track_len = len(track.findall("box"))
            self.assertEqual(track.get("id"), str(j+1))
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

        # checkout output video
        cap = cv2.VideoCapture(demo_path)
        video = cv2.VideoCapture(TestDetector2Cvat.video)
        self.assertTrue(cap.isOpened())
        self.assertTrue(video.isOpened())
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT),
                         video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                         video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                         video.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        video.release()

    @patch('kabr_tools.detector2cvat.YOLOv8')
    @patch('kabr_tools.detector2cvat.cv2.VideoCapture')
    def test_mock_with_data(self, video_capture, yolo):
        # mock outputs CVAT data
        ref_path = "tests/examples/MINISCENE1/metadata/DJI_tracks.xml"
        height, width, frames = 3078, 5472, 21
        data = DetectionData((height, width, 3),
                             frames,
                             ref_path)
        yolo_instance = MagicMock()
        yolo_instance.forward = data.forward
        yolo.return_value = yolo_instance
        yolo.get_centroid = MagicMock(
            side_effect=lambda pred: YOLOv8.get_centroid(pred))

        vc = MagicMock()
        vc.read = data.read
        vc.get = data.get
        video_capture.return_value = vc

        # run detector2cvat
        save = f"{self.save}/mock/1"
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", save]
        detector2cvat.main()

        # check output exists
        output_path = f"{save}/{self.dir}/DJI_0068.xml"
        self.assertTrue(file_exists(output_path))
        demo_path = f"{save}/{self.dir}/DJI_0068_demo.mp4"
        self.assertTrue(file_exists(demo_path))

        # check output xml
        xml_content = etree.parse(output_path).getroot()
        self.assertEqual(xml_content.tag, "annotations")
        ref_content = etree.parse(ref_path).getroot()
        ref_track = list(ref_content.findall("track"))
        for j, track in enumerate(xml_content.findall("track")):
            self.assertEqual(track.get("id"), str(j+1))
            self.assertEqual(track.get("label"), "Grevy")
            self.assertEqual(track.get("source"), "manual")
            ref_box = list(ref_track[j].findall("box"))
            for i, box in enumerate(track.findall("box")):
                self.assertEqual(box.get("frame"), ref_box[i].get("frame"))
                self.assertEqual(box.get("xtl"), ref_box[i].get("xtl"))
                self.assertEqual(box.get("ytl"), ref_box[i].get("ytl"))
                self.assertEqual(box.get("xbr"), ref_box[i].get("xbr"))
                self.assertEqual(box.get("ybr"), ref_box[i].get("ybr"))
                self.assertEqual(box.get("occluded"), "0")
                self.assertEqual(box.get("keyframe"), "1")
                self.assertEqual(box.get("z_order"), "0")
                # tracker marks last box as outside
                if i == frames - 1:
                    self.assertEqual(box.get("outside"), "1")
                else:
                    self.assertEqual(box.get("outside"), "0")

        # checkout output video
        cap = cv2.VideoCapture(demo_path)
        self.assertTrue(cap.isOpened())
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT),
                         frames)
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                         height)
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                         width)
        cap.release()

    @patch('kabr_tools.detector2cvat.YOLOv8')
    @patch('kabr_tools.detector2cvat.cv2.VideoCapture')
    def test_mock_noncontiguous(self, video_capture, yolo):
        # mock outputs non-contiguous frame detections
        ref_path = "tests/examples/DETECTOR1/DJI_tracks.xml"
        height, width, frames = 3078, 5472, 31
        data = DetectionData((height, width, 3),
                             frames,
                             ref_path)
        yolo_instance = MagicMock()
        yolo_instance.forward = data.forward
        yolo.return_value = yolo_instance
        yolo.get_centroid = MagicMock(
            side_effect=lambda pred: YOLOv8.get_centroid(pred))

        vc = MagicMock()
        vc.read = data.read
        vc.get = data.get
        video_capture.return_value = vc

        # run detector2cvat
        save = f"{self.save}/mock/2"
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", save]
        detector2cvat.main()

        # check output exists
        output_path = f"{save}/{self.dir}/DJI_0068.xml"
        self.assertTrue(file_exists(output_path))
        demo_path = f"{save}/{self.dir}/DJI_0068_demo.mp4"
        self.assertTrue(file_exists(demo_path))

        # check output xml
        xml_content = etree.parse(output_path).getroot()
        self.assertEqual(xml_content.tag, "annotations")
        ref_content = etree.parse(ref_path).getroot()
        ref_track = list(ref_content.findall("track"))
        for j, track in enumerate(xml_content.findall("track")):
            self.assertEqual(track.get("id"), str(j+1))
            self.assertEqual(track.get("label"), "Grevy")
            self.assertEqual(track.get("source"), "manual")
            ref_box = list(ref_track[j].findall("box"))
            i = 0
            frame = int(track.find("box").get("frame"))
            for box in track.findall("box"):
                if box.get("frame") == ref_box[i+1].get("frame"):
                    i += 1
                print(box.get("frame"), ref_box[i].get("frame"))
                self.assertEqual(box.get("frame"), str(frame))
                self.assertEqual(box.get("xtl"), ref_box[i].get("xtl"))
                self.assertEqual(box.get("ytl"), ref_box[i].get("ytl"))
                self.assertEqual(box.get("xbr"), ref_box[i].get("xbr"))
                self.assertEqual(box.get("ybr"), ref_box[i].get("ybr"))
                self.assertEqual(box.get("occluded"), "0")
                if box.get("frame") == ref_box[i].get("frame"):
                    self.assertEqual(box.get("keyframe"), "1")
                else:
                    self.assertEqual(box.get("keyframe"), "0")
                self.assertEqual(box.get("z_order"), "0")
                # tracker marks last box as outside
                if frame == frames - 1:
                    self.assertEqual(box.get("outside"), "1")
                else:
                    self.assertEqual(box.get("outside"), "0")
                frame += 1

        # checkout output video
        cap = cv2.VideoCapture(demo_path)
        self.assertTrue(cap.isOpened())
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT),
                         frames)
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                         height)
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                         width)
        cap.release()

    def test_parse_arg_min(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", self.save]
        args = detector2cvat.parse_args()

        # check parsed argument values
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.save, self.save)
        self.assertEqual(args.imshow, False)

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--save", self.save,
                    "--imshow"]
        args = detector2cvat.parse_args()

        # check parsed argument values
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.save, self.save)
        self.assertEqual(args.imshow, True)
