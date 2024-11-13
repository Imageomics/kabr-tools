import os
import shutil


def del_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def del_file(path):
    if os.path.exists(path):
        os.remove(path)
