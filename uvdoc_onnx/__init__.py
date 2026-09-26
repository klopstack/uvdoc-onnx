"""uvdoc-onnx — UVDoc document dewarping via ONNX Runtime (no torch).

Public API:
    from uvdoc_onnx import dewarp, DEFAULT_MODEL_PATH, DewarpError

    out, changed, error = dewarp(pil_image)            # bundled weights
    out, changed, error = dewarp(pil_image, model_path)  # custom weights
"""
from .core import DEFAULT_MODEL_PATH, DewarpError, dewarp

__version__ = "0.1.0"
__all__ = ["dewarp", "DEFAULT_MODEL_PATH", "DewarpError", "__version__"]
