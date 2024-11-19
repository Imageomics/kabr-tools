import unittest
import zipfile
import sys
import os
import requests
from unittest.mock import Mock, patch
import torch
import numpy as np
import pandas as pd
from kabr_tools import (
    miniscene2behavior,
    tracks_extractor
)
from kabr_tools.miniscene2behavior import annotate_miniscene
from tests.utils import del_file


TESTSDIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLESDIR = os.path.join(TESTSDIR, "examples")


def run():
    miniscene2behavior.main()


class TestMiniscene2Behavior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = "checkpoint_epoch_00075.pyth"
        # Download the model from Imageomics HF before running tests
        #cls.download_model()

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
        # Remove model files after all tests have been completed
        if os.path.exists(f"{cls.checkpoint}.zip"):
            os.remove(f"{cls.checkpoint}.zip")
        if os.path.exists(cls.checkpoint):
            os.remove(cls.checkpoint)

    def setUp(self):
        self.tool = "miniscene2behavior.py"
        self.hub = "zhong-al/x3d"
        self.checkpoint = "checkpoint_epoch_00075.pyth"
        self.miniscene = "mini-scenes/tests|detection_example|DJI_0068"
        self.video = "DJI_0068"
        self.config = "config.yml"
        self.gpu_num = "1"
        self.output = "DJI_0068.csv"

    def tearDown(self):
        # TODO: delete outputs
        # del_file(self.output)
        pass

    def test_run(self):
        # run tracks_extractor
        sys.argv = ["tracks_extractor.py",
                    "--video", "tests/detection_example/DJI_0068.mp4",
                    "--annotation", "tests/detection_example/DJI_0068.xml"]
        tracks_extractor.main()

        # annotate mini-scenes
        sys.argv = [self.tool,
                    "--hub", self.hub,
                    "--checkpoint", self.checkpoint,
                    "--miniscene", self.miniscene,
                    "--video", self.video]
        run()

    @patch('kabr_tools.utils.slowfast.utils.process_cv2_inputs')
    @patch('kabr_tools.utils.slowfast.utils.cv2.VideoCapture')
    def test_matching_tracks(self, video_capture, process_cv2_inputs):

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
        vc.get.return_value = 1

        self.output = '/tmp/annotation_data.csv'

        annotate_miniscene(cfg=mock_config,
                           model=mock_model,
                           miniscene_path=os.path.join(
                               EXAMPLESDIR, "MINISCENE1"),
                           video='DJI',
                           output_path=self.output)

        # Read in output CSV and make sure we have the expected columns and at least one row
        df = pd.read_csv(self.output, sep=' ')
        self.assertEqual(list(df.columns), [
                         "video", "track", "frame", "label"])
        self.assertGreater(len(df.index), 0)

    @patch('kabr_tools.utils.slowfast.utils.process_cv2_inputs')
    @patch('kabr_tools.utils.slowfast.utils.cv2.VideoCapture')
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
        vc.get.return_value = 1

        self.output = '/tmp/annotation_data.csv'

        annotate_miniscene(cfg=mock_config,
                           model=mock_model,
                           miniscene_path=os.path.join(
                               EXAMPLESDIR, "MINISCENE2"),
                           video='DJI',
                           output_path=self.output)

        # Read in output CSV and make sure we have the expected columns and at least one row
        df = pd.read_csv(self.output, sep=' ')
        self.assertEqual(list(df.columns), [
                         "video", "track", "frame", "label"])
        self.assertGreater(len(df.index), 0)

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
