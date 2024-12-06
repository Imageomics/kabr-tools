import unittest
import sys
import os
from unittest.mock import patch, Mock
import json
from lxml import etree
import cv2
from kabr_tools import tracks_extractor
from kabr_tools.utils.tracker import Tracker
from kabr_tools.utils.detector import Detector
from kabr_tools.utils.utils import get_scene
from tests.utils import (
    get_detection,
    dir_exists,
    file_exists,
    del_dir,
    del_file
)

# TODO: make constants for kabr tools (copied values in tracks_extractor.py)
scene_width, scene_height = 400, 300


def run():
    tracks_extractor.main()


class TestTracksExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # download data
        cls.video, cls.annotation = get_detection()

    @classmethod
    def tearDownClass(cls):
        # delete data
        del_file(cls.video)
        del_file(cls.annotation)

    def setUp(self):
        # set params
        self.tool = "tracks_extractor.py"
        self.video = TestTracksExtractor.video
        self.annotation = TestTracksExtractor.annotation

        # remove output directory
        del_dir("mini-scenes")

    def tearDown(self):
        # remove output directory
        # del_dir("mini-scenes")
        pass

    def test_run(self):
        # run tracks_extractor
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--annotation", self.annotation]
        run()

        # check output exists
        mini_folder = os.path.splitext("|".join(self.video.split("/")[-3:]))[0]
        video_name = "DJI_0068"
        self.assertTrue(dir_exists(f"mini-scenes/{mini_folder}"))
        self.assertTrue(dir_exists(f"mini-scenes/{mini_folder}/actions"))
        self.assertTrue(dir_exists(f"mini-scenes/{mini_folder}/metadata"))
        self.assertTrue(file_exists(f"mini-scenes/{mini_folder}/0.mp4"))
        self.assertTrue(file_exists(f"mini-scenes/{mini_folder}/1.mp4"))
        self.assertTrue(file_exists(
            f"mini-scenes/{mini_folder}/{video_name}.mp4"))
        self.assertTrue(file_exists(
            f"mini-scenes/{mini_folder}/metadata/{video_name}_metadata.json"))
        self.assertTrue(file_exists(
            f"mini-scenes/{mini_folder}/metadata/{video_name}_tracks.xml"))
        self.assertTrue(file_exists(
            f"mini-scenes/{mini_folder}/metadata/{video_name}.jpg"))

        # check metadata.json
        root = etree.parse(self.annotation).getroot()
        tracks = {"main":
                  [-1] * int("".join(root.find("meta").find("task").find("size").itertext()))}
        for track in root.iterfind("track"):
            track_id = track.attrib["id"]
            tracks[track_id] = []
            for box in track.iter("box"):
                frame_id = int(box.attrib["frame"])
                tracks[track_id].append(frame_id)
                tracks["main"][frame_id] = frame_id

        colors = list(Tracker.colors_table.values())

        with open(f"mini-scenes/{mini_folder}/metadata/{video_name}_metadata.json",
                  "r", encoding="utf-8") as f:
            metadata = json.load(f)
            self.assertTrue("original" in metadata)
            self.assertTrue("tracks" in metadata)
            self.assertTrue("colors" in metadata)
            self.assertEqual(metadata["original"], self.video)
            self.assertEqual(metadata["tracks"]["main"], tracks["main"])
            self.assertEqual(metadata["tracks"]["0"], tracks["0"])
            self.assertEqual(metadata["tracks"]["1"], tracks["1"])
            self.assertEqual(metadata["colors"]["0"],
                             list(colors[0 % len(colors)]))
            self.assertEqual(metadata["colors"]["1"],
                             list(colors[1 % len(colors)]))

        # check tracks.xml
        with open(f"mini-scenes/{mini_folder}/metadata/{video_name}_tracks.xml",
                  "r", encoding="utf-8") as f:
            track_copy = f.read()

        with open(self.annotation, "r", encoding="utf-8") as f:
            track = f.read()

        self.assertEqual(track, track_copy)

        # check 0.mp4, 1.mp4
        root = etree.parse(self.annotation).getroot()
        xml_tracks = {}
        for track in root.findall("track"):
            track_id = track.attrib["id"]
            xml_tracks[track_id] = track
        self.assertEqual(xml_tracks.keys(), {"0", "1"})

        original = cv2.VideoCapture(self.video)
        self.assertTrue(original.isOpened())
        mock = Mock()

        for track_id, xml_track in xml_tracks.items():
            track = cv2.VideoCapture(
                f"mini-scenes/{mini_folder}/{track_id}.mp4")
            self.assertTrue(track.isOpened())

            for i, box in enumerate(xml_track.iter("box")):
                original.set(cv2.CAP_PROP_POS_FRAMES, int(box.attrib["frame"]))
                track.set(cv2.CAP_PROP_POS_FRAMES, i)
                original_returned, original_frame = original.read()
                track_returned, track_frame = track.read()

                self.assertTrue(original_returned)
                self.assertTrue(track_returned)

                mock.box = [int(float(box.attrib["xtl"])),
                            int(float(box.attrib["ytl"])),
                            int(float(box.attrib["xbr"])),
                            int(float(box.attrib["ybr"]))]
                mock.centroid = Detector.get_centroid(mock.box)
                original_frame = get_scene(
                    original_frame, mock, scene_width, scene_height)

                cv2.imshow("a", original_frame)
                cv2.waitKey(0)
                cv2.imshow("a", track_frame)
                cv2.waitKey(0)

                # encoding seems to add some noise to frames, allow for that
                self.assertTrue(
                    cv2.norm(original_frame - track_frame) < 1e6)
            track.release()

        # check DJI_0068.mp4

        original.release()

    def test_parse_arg_min(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--annotation", self.annotation]
        args = tracks_extractor.parse_args()

        # check parsed arguments
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.annotation, self.annotation)

        # check default arguments
        self.assertEqual(args.tracking, False)
        self.assertEqual(args.imshow, False)

        # run tracks_extractor
        run()

    @patch('kabr_tools.tracks_extractor.cv2.imshow')
    def test_parse_arg_full(self, imshow):
        # parse arguments
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--annotation", self.annotation,
                    "--tracking",
                    "--imshow"]
        args = tracks_extractor.parse_args()

        # check parsed arguments
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.annotation, self.annotation)
        self.assertEqual(args.tracking, True)
        self.assertEqual(args.imshow, True)

        # run tracks_extractor
        run()
