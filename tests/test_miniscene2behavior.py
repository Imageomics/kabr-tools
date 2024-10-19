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


TESTSDIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLESDIR = os.path.join(TESTSDIR, "examples")


class TestMiniscene2Behavior(unittest.TestCase):
    def test_run(self):
        # run tracks_extractor
        sys.argv = ["tracks_extractor.py",
                    "--video", "tests/detection_example/DJI_0068.mp4",
                    "--annotation", "tests/detection_example/DJI_0068.xml"]
        tracks_extractor.main()

        # download model from huggingface
        url = "https://huggingface.co/imageomics/" \
            + "x3d-kabr-kinetics/resolve/main/" \
            + "checkpoint_epoch_00075.pyth.zip"
        r = requests.get(url, allow_redirects=True, timeout=300)
        with open("checkpoint_epoch_00075.pyth.zip", "wb") as f:
            f.write(r.content)

        # unzip model checkpoint
        with zipfile.ZipFile("checkpoint_epoch_00075.pyth.zip", "r") as zip_ref:
            zip_ref.extractall(".")

        # annotate mini-scenes
        sys.argv = ["miniscene2behavior.py",
                    "--checkpoint", "checkpoint_epoch_00075.pyth",
                    "--miniscene", "mini-scenes/tests|detection_example|DJI_0068",
                    "--video", "DJI_0068",]
        miniscene2behavior.main()

    @patch('kabr_tools.miniscene2behavior.process_cv2_inputs')
    @patch('kabr_tools.miniscene2behavior.cv2.VideoCapture')
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

        output_csv = '/tmp/annotation_data.csv'

        annotate_miniscene(cfg=mock_config,
                           model=mock_model,
                           miniscene_path=os.path.join(
                               EXAMPLESDIR, "MINISCENE1"),
                           video='DJI',
                           output_path=output_csv)

        # Read in output CSV and make sure we have the expected columns and at least one row
        df = pd.read_csv(output_csv, sep=' ')
        self.assertEqual(list(df.columns), [
                         "video", "track", "frame", "label"])
        self.assertGreater(len(df.index), 0)


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
        vc.get.return_value = 1

        output_csv = '/tmp/annotation_data.csv'

        annotate_miniscene(cfg=mock_config,
                           model=mock_model,
                           miniscene_path=os.path.join(
                               EXAMPLESDIR, "MINISCENE2"),
                           video='DJI',
                           output_path=output_csv)

        # Read in output CSV and make sure we have the expected columns and at least one row
        df = pd.read_csv(output_csv, sep=' ')
        self.assertEqual(list(df.columns), [
                         "video", "track", "frame", "label"])
        self.assertGreater(len(df.index), 0)

    def test_parse_arg_min(self):
        # parse arguments
        sys.argv = ["miniscene2behavior.py",
                    "--checkpoint", "checkpoint_epoch_00075.pyth",
                    "--miniscene", "mini-scenes/tests|detection_example|DJI_0068",
                    "--video", "DJI_0068"]
        args = miniscene2behavior.parse_args()
        self.assertEqual(args.config, "config.yml")
        self.assertEqual(args.checkpoint, "checkpoint_epoch_00075.pyth")
        self.assertEqual(args.gpu_num, 0)
        self.assertEqual(args.miniscene, "mini-scenes/tests|detection_example|DJI_0068")
        self.assertEqual(args.video, "DJI_0068")
        self.assertEqual(args.output, "annotation_data.csv")

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = ["miniscene2behavior.py",
                    "--config", "special_config.yml",
                    "--checkpoint", "checkpoint_epoch_00075.pyth",
                    "--gpu_num", "1",
                    "--miniscene", "mini-scenes/tests|detection_example|DJI_0068",
                    "--video", "DJI_0068",
                    "--output", "DJI_0068.csv"]
        args = miniscene2behavior.parse_args()
        self.assertEqual(args.config, "special_config.yml")
        self.assertEqual(args.checkpoint, "checkpoint_epoch_00075.pyth")
        self.assertEqual(args.gpu_num, 1)
        self.assertEqual(args.miniscene, "mini-scenes/tests|detection_example|DJI_0068")
        self.assertEqual(args.video, "DJI_0068")
        self.assertEqual(args.output, "DJI_0068.csv")
