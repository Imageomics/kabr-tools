import unittest
import zipfile
import sys
import os
from unittest.mock import Mock, patch
import requests
import torch
from lxml import etree
import numpy as np
import pandas as pd
from kabr_tools import (
    miniscene2behavior,
    tracks_extractor
)
from kabr_tools.miniscene2behavior import (
    create_model,
    annotate_miniscene,
    extract_config
)
from tests.utils import (
    del_file,
    del_dir,
    file_exists,
    same_path,
    get_detection,
    csv_equal
)


TESTSDIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLESDIR = os.path.join(TESTSDIR, "examples")
TEMPHUB = "zhong-al/x3d"


def run():
    miniscene2behavior.main()


class TestMiniscene2Behavior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # download the model from Imageomics HF
        cls.local_checkpoint = "checkpoint_epoch_00075.pyth"
        cls.download_model()

        # download data
        cls.video, cls.annotation = get_detection()

        # run tracks_extractor
        sys.argv = ["tracks_extractor.py",
                    "--video", cls.video,
                    "--annotation", cls.annotation]
        tracks_extractor.main()
        cls.miniscene = f'mini-scenes/{os.path.splitext("|".join(cls.video.split("/")[-3:]))[0]}'

    @classmethod
    def download_model(cls):
        if not os.path.exists(cls.local_checkpoint):
            url = "https://huggingface.co/imageomics/" \
                  + "x3d-kabr-kinetics/resolve/main/" \
                  + f"{cls.local_checkpoint}.zip"
            r = requests.get(url, allow_redirects=True, timeout=120)
            with open(f"{cls.local_checkpoint}.zip", "wb") as f:
                f.write(r.content)

            # Unzip model checkpoint
            with zipfile.ZipFile(f"{cls.local_checkpoint}.zip", "r") as zip_ref:
                zip_ref.extractall(".")

    @classmethod
    def tearDownClass(cls):
        # Remove model files after all tests have been completed
        if os.path.exists(f"{cls.local_checkpoint}.zip"):
            os.remove(f"{cls.local_checkpoint}.zip")
        if os.path.exists(cls.local_checkpoint):
            os.remove(cls.local_checkpoint)
        del_file(cls.video)
        del_file(cls.annotation)
        del_dir(cls.miniscene)

    def setUp(self):
        self.tool = "miniscene2behavior.py"
        self.hub = "imageomics/x3d-kabr-kinetics"
        self.checkpoint = "checkpoint_epoch_00075.pyth"
        self.checkpoint_archive = "checkpoint_epoch_00075.pyth.zip"
        self.miniscene = TestMiniscene2Behavior.miniscene
        self.video = "DJI_0068"
        self.config = "config.yml"
        self.test_config = "special_config.yml"
        self.gpu_num = "1"
        self.output = "DJI_0068.csv"
        self.example = "tests/examples"
        self.patch_index = [1]

    def tearDown(self):
        # delete output
        del_file(self.output)

# load model + run

    @patch("kabr_tools.miniscene2behavior.create_model")
    def test_hub_checkpoint_archive(self, create_mock):
        # patch create_model
        create_mock.side_effect = create_model

        # annotate mini-scenes
        sys.argv = [self.tool,
                    "--hub", self.hub,
                    "--checkpoint", self.checkpoint_archive,
                    "--miniscene", self.miniscene,
                    "--video", self.video,
                    "--output", self.output]
        run()

        # check arguments to create_model
        config_path = create_mock.call_args[0][0]
        checkpoint_path = create_mock.call_args[0][1]
        self.assertTrue(file_exists(config_path))
        self.assertTrue(file_exists(checkpoint_path))

        # check output
        self.assertTrue(csv_equal(self.output, f"{self.example}/{self.output}", self.patch_index))

    @patch("kabr_tools.miniscene2behavior.create_model")
    def test_hub_checkpoint(self, create_mock):
        # patch create_model
        create_mock.side_effect = create_model

        # annotate mini-scenes
        self.hub = TEMPHUB
        sys.argv = [self.tool,
                    "--hub", self.hub,
                    "--checkpoint", self.checkpoint,
                    "--miniscene", self.miniscene,
                    "--video", self.video,
                    "--output", self.output]
        run()

        # check arguments to create_model
        config_path = create_mock.call_args[0][0]
        checkpoint_path = create_mock.call_args[0][1]
        self.assertTrue(file_exists(config_path))
        self.assertTrue(file_exists(checkpoint_path))

        download_folder = f"{checkpoint_path.rsplit('/', 1)[0]}/"
        self.assertEqual(self.checkpoint,
                         checkpoint_path.replace(download_folder, ""))
        self.assertEqual(self.config,
                         config_path.replace(download_folder, ""))

        # check output
        self.assertTrue(csv_equal(self.output, f"{self.example}/{self.output}", self.patch_index))

    @patch("kabr_tools.miniscene2behavior.create_model")
    def test_hub_checkpoint_config(self, create_mock):
        # patch create_model
        create_mock.side_effect = create_model

        # annotate mini-scenes
        self.hub = TEMPHUB
        sys.argv = [self.tool,
                    "--hub", self.hub,
                    "--checkpoint", self.checkpoint,
                    "--config", self.config,
                    "--miniscene", self.miniscene,
                    "--video", self.video,
                    "--output", self.output]
        run()

        # check arguments to create_model
        config_path = create_mock.call_args[0][0]
        checkpoint_path = create_mock.call_args[0][1]
        self.assertTrue(file_exists(config_path))
        self.assertTrue(file_exists(checkpoint_path))

        download_folder = f"{checkpoint_path.rsplit('/', 1)[0]}/"
        self.assertEqual(self.checkpoint,
                         checkpoint_path.replace(download_folder, ""))
        self.assertEqual(self.config,
                         config_path.replace(download_folder, ""))

        # check output
        self.assertTrue(csv_equal(self.output, f"{self.example}/{self.output}", self.patch_index))

    @patch("kabr_tools.miniscene2behavior.create_model")
    def test_local_checkpoint(self, create_mock):
        # patch create_model
        create_mock.side_effect = create_model

        # download model
        self.download_model()

        # annotate mini-scenes
        sys.argv = [self.tool,
                    "--checkpoint", self.checkpoint,
                    "--miniscene", self.miniscene,
                    "--video", self.video,
                    "--output", self.output]
        run()

        # check arguments to create_model
        config_path = create_mock.call_args[0][0]
        checkpoint_path = create_mock.call_args[0][1]
        self.assertTrue(file_exists(config_path))
        self.assertTrue(file_exists(checkpoint_path))
        self.assertTrue(same_path(self.checkpoint, checkpoint_path))
        self.assertTrue(same_path(self.config, config_path))

        # check output
        self.assertTrue(csv_equal(self.output, f"{self.example}/{self.output}", self.patch_index))

    @patch("kabr_tools.miniscene2behavior.create_model")
    def test_local_checkpoint_config(self, create_mock):
        # patch create_model
        create_mock.side_effect = create_model

        # set args
        sys.argv = [self.tool,
                    "--checkpoint", self.checkpoint,
                    "--config", self.config,
                    "--miniscene", self.miniscene,
                    "--video", self.video,
                    "--output", self.output]
        args = miniscene2behavior.parse_args()

        # download model
        self.download_model()

        # extract config
        extract_config(args)

        # check args
        self.assertTrue(same_path(self.config, args.config))
        self.assertTrue(same_path(self.checkpoint, args.checkpoint))

        # annotate mini-scenes
        run()

        # check arguments to create_model
        config_path = create_mock.call_args[0][0]
        checkpoint_path = create_mock.call_args[0][1]
        self.assertTrue(file_exists(config_path))
        self.assertTrue(file_exists(checkpoint_path))
        self.assertTrue(same_path(self.checkpoint, checkpoint_path))
        self.assertTrue(same_path(self.config, config_path))

        # check output
        self.assertTrue(csv_equal(self.output, f"{self.example}/{self.output}", self.patch_index))

    def test_no_checkpoint(self):
        # annotate mini-scenes
        sys.argv = [self.tool,
                    "--miniscene", self.miniscene,
                    "--video", self.video]

        with self.assertRaises(SystemExit):
            run()

        with self.assertRaises(AssertionError):
            create_model(None, self.checkpoint, 0)

        with self.assertRaises(AssertionError):
            create_model(self.config, None, 0)

# output tracks

    @patch('kabr_tools.utils.slowfast.utils.get_input_clip')
    @patch('kabr_tools.utils.slowfast.utils.cv2.VideoCapture')
    def test_matching_tracks(self, video_capture, get_input_clip):
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

        root = etree.parse(
            f"{miniscene_dir}/metadata/DJI_tracks.xml").getroot()
        for track in root.iterfind("track"):
            track_id = int(track.get("id"))
            for box in track.iterfind("box"):
                row_val = [video_name, track_id, int(box.get("frame")), 98]
                self.assertEqual(list(df.loc[row_ct]), row_val)
                row_ct += 1
        self.assertEqual(len(df.index), row_ct)


    @patch('kabr_tools.miniscene2behavior.get_input_clip')
    @patch('kabr_tools.miniscene2behavior.cv2.VideoCapture')
    def test_nonmatching_tracks(self, video_capture, get_input_clip):

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

        root = etree.parse(
            f"{miniscene_dir}/metadata/DJI_tracks.xml").getroot()
        for track in root.iterfind("track"):
            track_id = int(track.get("id"))
            for box in track.iterfind("box"):
                row_val = [video_name, track_id, int(box.get("frame")), 0]
                self.assertEqual(list(df.loc[row_ct]), row_val)
                row_ct += 1
        self.assertEqual(len(df.index), row_ct)



# parse_args


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
        self.assertEqual(args.hub, None)
        self.assertEqual(args.config, None)
        self.assertEqual(args.gpu_num, 0)
        self.assertEqual(args.output, "annotation_data.csv")

    def test_parse_arg_full(self):
        # parse arguments
        sys.argv = [self.tool,
                    "--hub", self.hub,
                    "--config", self.test_config,
                    "--checkpoint", self.checkpoint,
                    "--gpu_num", self.gpu_num,
                    "--miniscene", self.miniscene,
                    "--video", self.video,
                    "--output", self.output]
        args = miniscene2behavior.parse_args()

        # check parsed argument values
        self.assertEqual(args.hub, self.hub)
        self.assertEqual(args.config, self.test_config)
        self.assertEqual(args.checkpoint, self.checkpoint)
        self.assertEqual(args.gpu_num, 1)
        self.assertEqual(args.miniscene, self.miniscene)
        self.assertEqual(args.video, self.video)
        self.assertEqual(args.output, self.output)
