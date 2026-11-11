"""SDFusion: conditional SDF estimation with selective state-space dynamics
and latent factor diffusion."""

from .data import Panel, load_panel
from .model import SDFusion
from .trainer import Trainer, run_rolling

__all__ = ["Panel", "load_panel", "SDFusion", "Trainer", "run_rolling"]
__version__ = "0.1.0"
