"""Isolated import bridge for the original EDBFormer source tree.

The released EDBFormer modules use absolute imports such as ``model.*`` and
``noise_layers.*``.  This bridge exposes the bundled, otherwise unmodified,
source tree without changing the model's internal implementation.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Tuple, Type

import torch.nn as nn


_CORE_ROOT = Path(__file__).resolve().parent / "EDBFormer_core"


def load_official_classes() -> Tuple[Type[nn.Module], Type[nn.Module]]:
    """Load the released 256x256 EDBFormer encoder and decoder classes."""
    if not _CORE_ROOT.is_dir():
        raise FileNotFoundError(
            f"EDBFormer core source directory was not found: {_CORE_ROOT}"
        )

    core_root = str(_CORE_ROOT)
    if core_root not in sys.path:
        sys.path.insert(0, core_root)

    # Guard against silently importing an unrelated top-level package called
    # `model`, which would otherwise produce confusing missing-module errors.
    loaded_model = sys.modules.get("model")
    if loaded_model is not None:
        loaded_path = Path(getattr(loaded_model, "__file__", "")).resolve()
        expected_dir = (_CORE_ROOT / "model").resolve()
        if expected_dir not in loaded_path.parents and loaded_path != expected_dir:
            raise RuntimeError(
                "A different top-level package named 'model' is already loaded "
                f"from {loaded_path}. Import EDBFormer before that package, or "
                "run it in a separate process."
            )

    encoder_module = importlib.import_module("model.cvt_encoder_2")
    decoder_module = importlib.import_module("model.cvt_decoder_2")
    return encoder_module.Encoder, decoder_module.Decoder
