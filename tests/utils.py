import os
import shutil
from huggingface_hub import hf_hub_download

DATA_HUB = "imageomics/kabr_testing"
REPO_TYPE = "dataset"

def get_hf(repo_id: str, filename: str, repo_type: str):
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)

def get_cached_datafile(filename: str):
    return get_hf(DATA_HUB, filename, REPO_TYPE)

def del_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def del_file(path):
    if os.path.exists(path):
        os.remove(path)
