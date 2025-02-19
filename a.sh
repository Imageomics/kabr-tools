#!/bin/bash

pip install -r requirements.txt
python -m pip install .
python -m unittest tests/test_detector2cvat.py
