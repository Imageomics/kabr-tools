import os
import shutil
from pathlib import Path


def del_dir(path):
    if dir_exists(path):
        shutil.rmtree(path)

def del_file(path):
    if file_exists(path):
        os.remove(path)

def file_exists(path):
    return Path(path).is_file()

def dir_exists(path):
    return Path(path).is_dir()

def same_path(path1, path2):
    return Path(path1).resolve() == Path(path2).resolve()
