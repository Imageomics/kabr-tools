import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

DATA_HUB = "imageomics/kabr_testing"
REPO_TYPE = "dataset"

DETECTION_VIDEO = "DJI_0068/DJI_0068.mp4"
DETECTION_ANNOTATION = "DJI_0068/DJI_0068.xml"

BEHAVIOR_VIDEO = "DJI_0001/DJI_0001.mp4"
BEHAVIOR_MINISCENE = "DJI_0001/43.mp4"
BEHAVIOR_ANNOTATION = "DJI_0001/actions/43.xml"
BEHAVIOR_METADATA = "DJI_0001/metadata/DJI_0001_metadata.json"

def get_hf(repo_id: str, filename: str, repo_type: str):
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)

def get_cached_datafile(filename: str):
    return get_hf(DATA_HUB, filename, REPO_TYPE)

def get_behavior():
    video = get_cached_datafile(BEHAVIOR_VIDEO)
    miniscene = get_cached_datafile(BEHAVIOR_MINISCENE)
    annotation = get_cached_datafile(BEHAVIOR_ANNOTATION)
    metadata = get_cached_datafile(BEHAVIOR_METADATA)
    return video, miniscene, annotation, metadata

def get_detection():    
    video = get_cached_datafile(DETECTION_VIDEO)
    annotation = get_cached_datafile(DETECTION_ANNOTATION)
    return video, annotation

def clean_dir(path):
    if os.path.exists(path):
        os.removedirs(path)

def del_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def del_file(path):
    if os.path.exists(path):
        os.remove(path)

def file_exists(path):
    return Path(path).is_file()


def dir_exists(path):
    return Path(path).is_dir()
