import os
import shutil
from pathlib import Path


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
