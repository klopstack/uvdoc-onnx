"""UVDoc neural dewarp — onnxruntime, CPU, no torch.

The model is UVDoc (Verhoeven et al., SIGGRAPH Asia 2023, MIT license,
github.com/tanguymagne/UVDoc): a ResNet-style encoder that predicts, for a
regular output grid, where each point's content comes from in the input
image (a 2D deformation grid in grid_sample coordinates). Bilinear
upsampling of the grid + grid-sampling the original image yields the
unwarped page. Trained on crumpled paper — this is the engine for
crumpled receipts, which planar quad/rectify approaches cannot handle.

License chain (why this package exists):
  * UVDoc itself is MIT — the weights and architecture are freely
    redistributable.
  * The upstream repo ships a PyTorch demo only (``demo.py`` +
    ``model/best_model.pkl``), no ONNX export.
  * The ``docuwarp`` PyPI package wraps the same model but ships NO
    license (citation only) and is dormant.
  * So: we export the MIT model to ONNX once (``scripts/export_uvdoc_onnx.py``,
    parity-checked against the PyTorch demo) and ship the artifact here.
    Runtime needs onnxruntime only — NO torch.

Model contract (from upstream ``demo.py`` / ``utils.py``):
    input   (1, 3, 488, 712) float32, RGB, /255, resized (aspect squish
             is the model's training condition — portrait documents)
    output  (1, 2, 45, 31) float32 — 2D grid (Gh=45, Gw=31), values in
             [-1, 1]; channel 0 = x, channel 1 = y
    remap   grid bilinear-upscaled to (H, W) with align_corners=True,
            then grid_sample(original, grid, align_corners=True,
            border zero)
  * The 3D-position output head is discarded (the demo uses only the 2D
    grid for remapping).

Failure policy: dewarp is a QUALITY stage, never a correctness gate. Any
error (missing model file, ORT failure) returns the input UNCHANGED plus
an error string — a document is never lost to a dewarp problem.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

_log = logging.getLogger("uvdoc-onnx")

# Bundled weights (committed to the repo — MIT, 214 KB).
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "uvdoc.onnx"

# Upstream constants (utils.py): model input size (w, h) and grid size.
# GRID_SIZE = [45, 31]; per bilinear_unwarping's docstring the output
# tensor is (B, 2, Gh, Gw) → Gh=45, Gw=31 (≈15.8 px/sample in BOTH
# directions on the 488×712 portrait input — the uniform sampling a
# trained model wants).
_IMG_SIZE_WH = (488, 712)
_GRID_SIZE_GHW = (45, 31)

# Wall-clock budget for one inference. ORT runs synchronously (we cannot
# safely interrupt a running inference), so this is a WARN threshold, not
# a hard kill — a genuinely hung session would be an ORT bug, and the
# exception path below covers crashes.
TIME_BUDGET_SECONDS = 120.0

_sessions: dict[str, object] = {}  # path → InferenceSession (lazy, cached)


class DewarpError(RuntimeError):
    pass


def _get_session(model_path: str):
    """Load (once) and cache the ORT session for ``model_path``."""
    sess = _sessions.get(model_path)
    if sess is not None:
        return sess
    import onnxruntime as ort
    if not Path(model_path).is_file():
        raise DewarpError(
            f"model not found: {model_path} "
            "(run scripts/export_uvdoc_onnx.py to produce one)")
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 8
    t0 = time.monotonic()
    sess = ort.InferenceSession(model_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    _log.info("UVDoc session loaded from %s in %.1fs", model_path,
              time.monotonic() - t0)
    _sessions[model_path] = sess
    return sess


def _upsample_grid(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Replicate torch F.interpolate(mode='bilinear', align_corners=True).

    grid: (2, Gh, Gw) in [-1, 1] → (2, out_h, out_w). align_corners=True
    maps output pixel i to input coordinate i·(in−1)/(out−1) — a plain
    linspace, which is what makes this exact (cv2.resize uses a different,
    pixel-centred mapping and would drift by up to half a grid cell).
    """
    gh, gw = grid.shape[1:]
    ys = np.linspace(0.0, gh - 1, out_h)
    xs = np.linspace(0.0, gw - 1, out_w)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, gh - 1)
    x1 = np.minimum(x0 + 1, gw - 1)
    wy = (ys - y0)[:, None]  # (out_h, 1)
    wx = (xs - x0)[None, :]  # (1, out_w)
    out = np.empty((2, out_h, out_w), dtype=np.float32)
    for c in range(2):
        g = grid[c]
        top = g[y0][:, x0] * (1 - wx) + g[y0][:, x1] * wx
        bot = g[y1][:, x0] * (1 - wx) + g[y1][:, x1] * wx
        out[c] = top * (1 - wy) + bot * wy
    return out


def _grid_sample(img: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Replicate torch F.grid_sample(align_corners=True, border zero).

    img: (H, W, 3) float32 in [0, 1]; grid: (2, H, W) in [-1, 1]
    (channel 0 = x, channel 1 = y; -1 = left/top edge, +1 = right/bottom).
    Out-of-bounds samples are ZERO (grid_sample's default padding).
    """
    h, w = img.shape[:2]
    gx, gy = grid[0], grid[1]
    px = (gx + 1.0) * (w - 1) / 2.0
    py = (gy + 1.0) * (h - 1) / 2.0
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    # Boundary-INCLUSIVE, matching torch: a sample exactly at the edge
    # (px == w-1, i.e. gx == +1) is in-bounds and hits the last pixel.
    # (Checking x1 < w instead would zero the last row/column of the page.)
    in_bounds = (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
    xc = np.clip(x0, 0, w - 1)
    xc1 = np.clip(x1, 0, w - 1)
    yc = np.clip(y0, 0, h - 1)
    yc1 = np.clip(y1, 0, h - 1)
    wx = (px - x0)[:, :, None]
    wy = (py - y0)[:, :, None]
    top = img[yc, xc] * (1 - wx) + img[yc, xc1] * wx
    bot = img[yc1, xc] * (1 - wx) + img[yc1, xc1] * wx
    out = (top * (1 - wy) + bot * wy).astype(np.float32)
    out[~in_bounds] = 0.0
    return out


def dewarp(img: Image.Image,
           model_path: str | Path | None = None,
           log: logging.Logger | None = None
           ) -> tuple[Image.Image, bool, str | None]:
    """Unwarp one document photo with UVDoc.

    Args:
        img: PIL image of a (roughly portrait) document photo.
        model_path: ONNX weights; defaults to the bundled
            ``uvdoc_onnx/models/uvdoc.onnx``.
        log: optional logger override.

    Returns ``(result, changed, error)``:
      * success → (unwarped image, True, None)
      * any failure → (INPUT unchanged, False, error string)

    The caller decides what to do with ``error`` (typically: flag the
    document and continue with the uncorrected image).
    """
    logger = log if log is not None else _log
    model_path = str(model_path or DEFAULT_MODEL_PATH)
    try:
        sess = _get_session(model_path)
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        h, w = rgb.shape[:2]
        # Resize to the model's fixed input (w, h) — cv2 takes (width, height).
        import cv2
        small = cv2.resize(rgb, _IMG_SIZE_WH, interpolation=cv2.INTER_AREA)
        inp = small.transpose(2, 0, 1)[None, ...].astype(np.float32)

        t0 = time.monotonic()
        outputs = sess.run(None, {sess.get_inputs()[0].name: inp})
        elapsed = time.monotonic() - t0
        if elapsed > TIME_BUDGET_SECONDS:
            logger.warning("UVDoc inference took %.1fs (> %.0fs budget)",
                           elapsed, TIME_BUDGET_SECONDS)
        grid = outputs[0][0]  # (2, Gh, Gw)
        if grid.shape != (2, *_GRID_SIZE_GHW):
            raise DewarpError(f"unexpected grid shape {grid.shape}")

        full_grid = _upsample_grid(grid, h, w)
        unwarped = (_grid_sample(rgb, full_grid) * 255.0)
        return Image.fromarray(unwarped.astype(np.uint8), mode="RGB"), True, None
    except Exception as err:  # noqa: BLE001 — quality stage, never fatal
        logger.warning("UVDoc dewarp failed: %s", err)
        return img, False, str(err)
