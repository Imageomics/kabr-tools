import unittest
import requests
import zipfile
import sys
from kabr_tools import (
    miniscene2behavior,
    tracks_extractor
    )


class TestMiniscene2Behavior(unittest.TestCase):
    def test_annotate(self):
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
