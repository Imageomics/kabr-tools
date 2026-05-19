import os
import shutil
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

#DATA_HUB = "imageomics/kabr_testing"
#REPO_TYPE = "dataset"

BASE = "kabr-tools/tests/dataset"

DETECTION_VIDEO = BASE + "/DJI_0068/DJI_0068.mp4" #DJI 68 - mp4 
DETECTION_ANNOTATION = BASE + "/DJI_0068/DJI_0068.xml" #DJI 68 - xml

BEHAVIOR_VIDEO = BASE + "/DJI_0001/DJI_0001.mp4" #DJI 1 - mp4 
BEHAVIOR_MINISCENE = BASE + "/43/43.mp4" #DJI 1 - 43 - mp4 
BEHAVIOR_ANNOTATION = BASE + "/43/43.xml" #DJI 1 - 43 - xml 
BEHAVIOR_METADATA = BASE + "/DJI_0001/DJI_0001_metadata.json" #DJI 1 - metadata - json 

'''
def get_hf(repo_id: str, filename: str, repo_type: str):
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def get_cached_datafile(filename: str):
    return get_hf(DATA_HUB, filename, REPO_TYPE)
'''

def get_behavior():
    video = BEHAVIOR_VIDEO
    miniscene = BEHAVIOR_MINISCENE
    annotation = BEHAVIOR_ANNOTATION
    metadata = BEHAVIOR_METADATA
    return video, miniscene, annotation, metadata


def get_detection():
    video = DETECTION_VIDEO
    annotation = DETECTION_ANNOTATION
    return video, annotation


def clean_empty_dirs(path):
    """remove the empty parent directories alongside path directory
    for removing the empty nested directories created when
    using slowfast model"""
    if os.path.exists(path) and len(os.listdir(path)) == 0:
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


def same_path(path1, path2):
    return Path(path1).resolve() == Path(path2).resolve()


def csv_equal(path1, path2, acceptable_diff=None):
    df1 = pd.read_csv(path1, sep=" ")
    df2 = pd.read_csv(path2, sep=" ")

    if not acceptable_diff:
        acceptable_diff = []

    if not df1.index.equals(df2.index):
        return False

    diffs = []
    for ind in df1.index:
        if not df1.loc[ind].equals(df2.loc[ind]):
            diffs.append(ind)

    return df1.equals(df2) or set(diffs).issubset(acceptable_diff)
