# TODO: get rid of if False when model is public
import unittest
import requests
import zipfile
from time import sleep
import sys
from kabr_tools import miniscene2behavior


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
