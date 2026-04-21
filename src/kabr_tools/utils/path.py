import os
from pathlib import Path


def join_paths(*parts: str | os.PathLike[str]) -> str:
    """Join one or more path components and return the result as a string."""
    if not parts:
        raise ValueError("At least one path must be provided")
    return str(Path(*parts))
