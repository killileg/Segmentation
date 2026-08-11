"""Pipeline implementations, selected by dimensionality."""

from .base import BasePipeline
from .pipeline2d import Pipeline2D
from .pipeline3d import Pipeline3D

PIPELINES = {"2d": Pipeline2D, "3d": Pipeline3D}


def get_pipeline(dim: str):
    """Return the pipeline class for ``"2d"`` or ``"3d"``."""
    key = dim.lower()
    if key not in PIPELINES:
        raise KeyError(f"unknown dimensionality {dim!r}; expected one of {sorted(PIPELINES)}")
    return PIPELINES[key]


__all__ = ["BasePipeline", "Pipeline2D", "Pipeline3D", "PIPELINES", "get_pipeline"]
