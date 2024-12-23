from pathlib import Path


def join_paths(parts):
    return str(Path(*parts))
