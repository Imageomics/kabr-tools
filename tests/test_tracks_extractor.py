import unittest
import sys
import os
import json
from lxml import etree
from kabr_tools import tracks_extractor
from kabr_tools.utils.tracker import Tracker


class TestTracksExtractor(unittest.TestCase):
    def setUp(self):
        self.tool = "tracks_extractor.py"
        self.video = "tests/detection_example/DJI_0068.mp4"
        self.annotation = "tests/detection_example/DJI_0068.xml"

    def test_run(self):
        # run tracks_extractor
        sys.argv = [self.tool,
                    "--video", self.video,
                    "--annotation", self.annotation]
        tracks_extractor.main()

        # check output exists
        mini_folder = "tests|detection_example|DJI_0068"
        video_name = "DJI_0068"
        self.assertTrue(os.path.exists(f"mini-scenes/{mini_folder}"))
        self.assertTrue(os.path.exists(f"mini-scenes/{mini_folder}/actions"))
        self.assertTrue(os.path.exists(f"mini-scenes/{mini_folder}/metadata"))
        self.assertTrue(os.path.exists(f"mini-scenes/{mini_folder}/0.mp4"))
        self.assertTrue(os.path.exists(f"mini-scenes/{mini_folder}/1.mp4"))
        self.assertTrue(os.path.exists(
            f"mini-scenes/{mini_folder}/{video_name}.mp4"))
        self.assertTrue(os.path.exists(
            f"mini-scenes/{mini_folder}/metadata/{video_name}_metadata.json"))
        self.assertTrue(os.path.exists(
            f"mini-scenes/{mini_folder}/metadata/{video_name}_tracks.xml"))
        self.assertTrue(os.path.exists(
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

        # TODO: check 0.mp4, 1.mp4, DJI_0068.mp4

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

    def test_parse_arg_full(self):
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
