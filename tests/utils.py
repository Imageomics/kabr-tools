import os
import shutil
from pathlib import Path
import pandas as pd


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


def csv_equal(path1, path2):
    df1 = pd.read_csv(path1, sep="")
    df2 = pd.read_csv(path2, sep="")

    return df1.equals(df2)
