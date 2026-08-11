"""msfusion — multimodal X-ray / neutron clast segmentation.

A comparison of fusion *strategies* (no fusion, early, late, learned stacking) with the backbone
architecture held fixed, plus an SSFB-backbone arm that isolates the architecture's own
contribution. See the README for the full method table.
"""

__version__ = "0.1.0"

from .config import Config, DataConfig, MetaConfig, PathConfig, SplitConfig, TrainConfig
from .methods import BRANCHES, METHODS, required_branches, resolve_methods

__all__ = [
    "Config",
    "PathConfig",
    "DataConfig",
    "SplitConfig",
    "TrainConfig",
    "MetaConfig",
    "METHODS",
    "BRANCHES",
    "resolve_methods",
    "required_branches",
    "__version__",
]
