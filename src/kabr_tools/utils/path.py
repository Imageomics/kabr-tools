from pathlib import Path


def join_paths(*parts):
    assert len(parts) > 0, "At least one path must be provided"
    return str(Path(*parts))
