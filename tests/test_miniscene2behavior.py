import unittest
import zipfile
import sys
import os
import requests
from unittest.mock import Mock, patch
from lxml import etree
import torch
import numpy as np
import pandas as pd
from kabr_tools import (
    miniscene2behavior,
    tracks_extractor
)
from kabr_tools.miniscene2behavior import annotate_miniscene
from tests.utils import (
    del_file,
    del_dir,
    get_detection
)


TESTSDIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLESDIR = os.path.join(TESTSDIR, "examples")


def run():
    miniscene2behavior.main()


class TestMiniscene2Behavior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # download the model from Imageomics HF
        cls.checkpoint = "checkpoint_epoch_00075.pyth"
        cls.download_model()

        # download data
        cls.video, cls.annotation = get_detection()

        # extract mini-scene
        sys.argv = ["tracks_extractor.py",
                    "--video", cls.video,
                    "--annotation", cls.annotation]
        tracks_extractor.main()
        cls.miniscene = f'mini-scenes/{os.path.splitext("|".join(cls.video.split("/")[-3:]))[0]}'

    @classmethod
    def download_model(cls):
        if not os.path.exists(cls.checkpoint):
            url = "https://huggingface.co/imageomics/" \
                  + "x3d-kabr-kinetics/resolve/main/" \
                  + f"{cls.checkpoint}.zip"
            r = requests.get(url, allow_redirects=True, timeout=120)
            with open(f"{cls.checkpoint}.zip", "wb") as f:
                f.write(r.content)

            # unzip model checkpoint
            with zipfile.ZipFile(f"{cls.checkpoint}.zip", "r") as zip_ref:
                zip_ref.extractall(".")

    @classmethod
    def tearDownClass(cls):
        # remove model files after tests
        if os.path.exists(f"{cls.checkpoint}.zip"):
            os.remove(f"{cls.checkpoint}.zip")
        if os.path.exists(cls.checkpoint):
            os.remove(cls.checkpoint)
        del_file(cls.video)
        del_file(cls.annotation)
        del_dir(cls.miniscene)

    def setUp(self):
        self.tool = "miniscene2behavior.py"
        self.checkpoint = "checkpoint_epoch_00075.pyth"
        self.miniscene = TestMiniscene2Behavior.miniscene
        self.video = "DJI_0068"
        self.config = "special_config.yml"
        self.gpu_num = "1"
        self.output = "DJI_0068.csv"

    def tearDown(self):
        # delete outputs
        del_file(self.output)

    def test_run(self):
        # download model
        self.download_model()

        # annotate mini-scenes
        sys.argv = [self.tool,
                    "--checkpoint", self.checkpoint,
                    "--miniscene", self.miniscene,
                    "--video", self.video,
                    "--output", self.output]
        run()

        # check output CSV
        df = pd.read_csv(self.output, sep=' ')
        self.assertEqual(list(df.columns), [
                         "video", "track", "frame", "label"])
        row_ct = 0

        root = etree.parse(f"{self.miniscene}/metadata/{self.video}_tracks.xml").getroot()
        for track in root.iterfind("track"):
            track_id = int(track.get("id"))
            for box in track.iterfind("box"):
                row = list(df.loc[row_ct])
                self.assertEqual(row[0], self.video)
                self.assertEqual(row[1], track_id)
                self.assertEqual(row[2], int(box.get("frame")))
                self.assertTrue(row[3] >= 0)
                self.assertTrue(row[3] <= 7)
                row_ct += 1
        self.assertEqual(len(df.index), row_ct)

    @patch('kabr_tools.miniscene2behavior.process_cv2_inputs')
    @patch('kabr_tools.miniscene2behavior.cv2.VideoCapture')
    def test_matching_tracks(self, video_capture, process_cv2_inputs):
        # create fake model that weights class 98
        mock_model = Mock()
        prob = torch.zeros(99)
        prob[-1] = 1
        mock_model.return_value = prob

        # create fake cfg
        mock_config = Mock(
            DATA=Mock(NUM_FRAMES=16,
                      SAMPLING_RATE=5,
                      TEST_CROP_SIZE=300),
            NUM_GPUS=0,
            OUTPUT_DIR=''
        )

        # create fake video capture
        vc = video_capture.return_value
        vc.read.return_value = True, np.zeros((8, 8, 3), np.uint8)
        vc.get.return_value = 21

        self.output = '/tmp/annotation_data.csv'
        miniscene_dir = os.path.join(EXAMPLESDIR, "MINISCENE1")
        video_name = "DJI"

        annotate_miniscene(cfg=mock_config,
                           model=mock_model,
                           miniscene_path=miniscene_dir,
                           video=video_name,
                           output_path=self.output)

        # check output CSV
        df = pd.read_csv(self.output, sep=' ')
        self.assertEqual(list(df.columns), [
                         "video", "track", "frame", "label"])
        row_ct = 0

        root = etree.parse(f"{miniscene_dir}/metadata/DJI_tracks.xml").getroot()
        for track in root.iterfind("track"):
            track_id = int(track.get("id"))
            for box in track.iterfind("box"):
                row_val = [video_name, track_id, int(box.get("frame")), 98]
                self.assertEqual(list(df.loc[row_ct]), row_val)
                row_ct += 1
        self.assertEqual(len(df.index), row_ct)

    @patch('kabr_tools.miniscene2behavior.process_cv2_inputs')
    @patch('kabr_tools.miniscene2behavior.cv2.VideoCapture')
    def test_nonmatching_tracks(self, video_capture, process_cv2_inputs):

        # Create fake model that always returns a prediction of 1
        mock_model = Mock()
        mock_model.return_value = torch.tensor([1])

        # Create fake cfg
        mock_config = Mock(
            DATA=Mock(NUM_FRAMES=16,
                      SAMPLING_RATE=5,
                      TEST_CROP_SIZE=300),
            NUM_GPUS=0,
            OUTPUT_DIR=''
        )

        # Create fake video capture
        vc = video_capture.return_value
        vc.read.return_value = True, np.zeros((8, 8, 3), np.uint8)
        vc.get.return_value = 21

        self.output = '/tmp/annotation_data.csv'
        miniscene_dir = os.path.join(EXAMPLESDIR, "MINISCENE2")
        video_name = "DJI"

        annotate_miniscene(cfg=mock_config,
                           model=mock_model,
                           miniscene_path=os.path.join(
                               EXAMPLESDIR, "MINISCENE2"),
                           video='DJI',
                           output_path=self.output)

        # check output CSV
        df = pd.read_csv(self.output, sep=' ')
        self.assertEqual(list(df.columns), [
                         "video", "track", "frame", "label"])
        row_ct = 0

        root = etree.parse(f"{miniscene_dir}/metadata/DJI_tracks.xml").getroot()
        for track in root.iterfind("track"):
            track_id = int(track.get("id"))
            for box in track.iterfind("box"):
                row_val = [video_name, track_id, int(box.get("frame")), 0]
                self.assertEqual(list(df.loc[row_ct]), row_val)
                row_ct += 1
        self.assertEqual(len(df.index), row_ct)

    def test_parse_arg_min(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--checkpoint", self.checkpoint,
                    "--miniscene", self.miniscene,
                    "--video", self.video]
        args = miniscene2behavior.parse_args()

        # check parsed argument values
        self.assertEqual(args.checkpoint, self.checkpoint)
        self.assertEqual(args.miniscene, self.miniscene)
        self.assertEqual(args.video, self.video)

        # check default argument values
        self.assertEqual(args.config, "config.yml")
        self.assertEqual(args.gpu_num, 0)
        self.assertEqual(args.output, "annotation_data.csv")

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--config", self.config,
                    "--checkpoint", self.checkpoint,
                    "--gpu_num", self.gpu_num,
                    "--miniscene", self.miniscene,
                    "--video", self.video,
                    "--output", self.output]
        args = miniscene2behavior.parse_args()

        # check parsed argument values
        self.assertEqual(args.config, self.config)
        self.assertEqual(args.checkpoint, self.checkpoint)
        self.assertEqual(args.gpu_num, 1)
        self.assertEqual(args.miniscene, self.miniscene)
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.output, self.output)
