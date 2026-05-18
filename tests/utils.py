import os
import shutil
import tempfile
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_TYPE = "dataset"

BEHAVIOR_HUB = "imageomics/KABR-mini-scene-raw-videos"
DETECTION_VIDEO_HUB = "imageomics/KABR-raw-videos"
DETECTION_ANNOTATION_HUB = "imageomics/kabr-worked-examples"

BEHAVIOR_VIDEO = "16_01_23_flight_1-DJI_0001/16_01_23-DJI_0001-trimmed.MP4"
BEHAVIOR_MINISCENE = "16_01_23_flight_1-DJI_0001/43.mp4"
BEHAVIOR_ANNOTATION = "16_01_23_flight_1-DJI_0001/actions/43.xml"
BEHAVIOR_METADATA = "16_01_23_flight_1-DJI_0001/metadata/DJI_0001_metadata.json"

DETECTION_VIDEO = "18_01_2023_session_7/DJI_0068_trimmed.mp4"
DETECTION_ANNOTATION = "detections/18_01_2023_session_7-DJI_0068.xml"


def get_hf(repo_id: str, filename: str, repo_type: str):
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def get_behavior():
    video_hf = get_hf(BEHAVIOR_HUB, BEHAVIOR_VIDEO, REPO_TYPE)
    miniscene_hf = get_hf(BEHAVIOR_HUB, BEHAVIOR_MINISCENE, REPO_TYPE)
    annotation_hf = get_hf(BEHAVIOR_HUB, BEHAVIOR_ANNOTATION, REPO_TYPE)
    metadata_hf = get_hf(BEHAVIOR_HUB, BEHAVIOR_METADATA, REPO_TYPE)

    tmpdir = tempfile.mkdtemp()
    base = Path(tmpdir) / "DJI_0001"
    (base / "actions").mkdir(parents=True)
    (base / "metadata").mkdir(parents=True)

    video = base / "DJI_0001.mp4"
    miniscene = base / "43.mp4"
    annotation = base / "actions" / "43.xml"
    metadata = base / "metadata" / "DJI_0001_metadata.json"

    os.symlink(video_hf, video)
    os.symlink(miniscene_hf, miniscene)
    os.symlink(annotation_hf, annotation)
    shutil.copy2(metadata_hf, metadata)

    return str(video), str(miniscene), str(annotation), str(metadata)


def get_detection():
    video_hf = get_hf(DETECTION_VIDEO_HUB, DETECTION_VIDEO, REPO_TYPE)
    annotation_hf = get_hf(DETECTION_ANNOTATION_HUB, DETECTION_ANNOTATION, REPO_TYPE)

    tmpdir = tempfile.mkdtemp()
    base = Path(tmpdir) / "DJI_0068"
    base.mkdir()

    video = base / "DJI_0068.mp4"
    annotation = base / "DJI_0068.xml"

    os.symlink(video_hf, video)
    shutil.copy2(annotation_hf, annotation)

    return str(video), str(annotation)


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
