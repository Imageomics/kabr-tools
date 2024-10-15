# TODO: get rid of if False when model is public
import unittest
from unittest.mock import Mock, patch
import os
import requests
import zipfile
from time import sleep
import sys
from kabr_tools import miniscene2behavior
from kabr_tools.miniscene2behavior import annotate_miniscene
import torch
import pandas as pd

TESTSDIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLESDIR = os.path.join(TESTSDIR, "examples")


class TestMiniscene2Behavior(unittest.TestCase):
    def test_annotate(self):
        if False:
            # wait for tracks_extractor test
            sleep(5)

            # note: the following download request
            # requires model to be public

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
                        "--miniscene", "mini-scenes/tests|examples|DJI_0068",
                        "--video", "DJI_0068",]
            miniscene2behavior.main()

    # Replace process_cv2_inputs function with a mock
    @patch('kabr_tools.miniscene2behavior.process_cv2_inputs')
    def test_annotate_miniscene(self, process_cv2_inputs):

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

        output_csv = '/tmp/annotation_data.csv'

        annotate_miniscene(cfg=mock_config,
                           model=mock_model,
                           miniscene_path=os.path.join(EXAMPLESDIR, "MINISCENE1"),
                           video='DJI_0068',
                           output_path=output_csv)

        # Read in output CSV and make sure we have the expected columns and at least one row
        df = pd.read_csv(output_csv, sep=' ')
        self.assertEqual(list(df.columns), ["video","track","frame", "label"])
        self.assertGreater(len(df.index), 0)
