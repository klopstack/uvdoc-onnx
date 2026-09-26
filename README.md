# uvdoc-onnx

Document dewarping (flattening) for **crumpled, bent, and photographed
documents**, powered by [UVDoc](https://github.com/tanguymagne/UVDoc) and run
through **ONNX Runtime — no torch required at inference time**.

UVDoc (Verhoeven et al., SIGGRAPH Asia 2023) is a neural grid-based document
unwarping model trained on crumpled paper. It is the right tool for the case
planar approaches miss: a receipt that has been folded, crumpled, or shot at
an angle on a table. This package makes that model practical to ship:

- **One `pip install`** — the ONNX weights (214 KB) are bundled. No torch, no
  CUDA, no multi-GB model download.
- **A tiny, dependency-light runtime** — `onnxruntime` + `numpy` +
  `opencv-python-headless` + `pillow`. Runs on CPU in well under a second.
- **A clean license chain** — MIT model + MIT glue. (See *Why this exists*.)
- **A parity-checked export** — the bundled ONNX artifact is validated against
  the upstream PyTorch model before it is accepted.

## Install

```bash
pip install uvdoc-onnx
```

Or from source:

```bash
git clone https://github.com/klopstack/uvdoc-onnx
cd uvdoc-onnx
pip install -e .
```

## Usage

```python
from PIL import Image
from uvdoc_onnx import dewarp

img = Image.open("crumpled_receipt.jpg")
out, changed, error = dewarp(img)          # uses the bundled weights

if error:
    # Dewarp is a quality stage, never a correctness gate: on ANY failure the
    # input is returned unchanged plus an error string. Decide what to do
    # (flag it, fall back to the uncorrected image, …).
    print("dewarp unavailable:", error)
else:
    out.save("receipt_flattened.jpg")
```

`dewarp(img, model_path=None, log=None)` returns a 3-tuple:

| Return | Meaning |
| ------ | ------- |
| `result` | The unwarped PIL image on success; the **input unchanged** on any failure. |
| `changed` | `True` if a new image was produced, `False` on failure. |
| `error` | `None` on success, otherwise a human-readable error string. |

Pass a custom `model_path` to use different weights (e.g. a re-export, or a
fine-tune). The default is the bundled `uvdoc_onnx/models/uvdoc.onnx`.

### Where it fits

Typical photo-of-document pipeline:

1. **Planar rectify** (OpenCV quad detect + perspective warp) — for documents
   photographed at an angle but lying flat. Cheap, no model.
2. **Neural dewarp** (this package) — for bent/crumpled paper, and as the
   fallback when no planar quad is found (crumpled paper is non-planar).
3. **Deskew** (e.g. tesseract OSD) — small residual rotation.
4. **OCR** on the corrected image, so any text layer aligns with the
   transcription.

This package is step 2. Steps 1 and 3 are intentionally out of scope.

## How it works

The model predicts, for a regular output grid, where each point's content
comes from in the input image (a 2D deformation grid in `grid_sample`
coordinates). The runtime replicates the upstream remap exactly:

- input: `(1, 3, 488, 712)` float32, RGB, `/255`, aspect-squished to the
  model's fixed portrait input (that squish is the model's training
  condition);
- output: `(1, 2, 45, 31)` float32 grid in `[-1, 1]` (channel 0 = x,
  channel 1 = y);
- remap: bilinear-upsample the grid to `(H, W)` with `align_corners=True`,
  then `grid_sample` the original image (`align_corners=True`, zero padding,
  **boundary-inclusive** so the last row/column isn't zeroed).

The 3D-position output head is discarded (the upstream demo uses only the 2D
grid for remapping).

## Regenerating the weights

The bundled ONNX was exported from the upstream PyTorch checkpoint and
**parity-checked** against it on the upstream example images (max grid diff
must be ≤ 1e-3). To reproduce:

```bash
python scripts/export_uvdoc_onnx.py            # → uvdoc_onnx/models/uvdoc.onnx
python scripts/export_uvdoc_onnx.py out.onnx   # → custom path
```

This clones the upstream repo, builds a throwaway torch-CPU venv, exports at
opset 18, validates parity, and atomically swaps the artifact in (keeping the
previous one as `.bak`). It needs `git` and a few hundred MB of bandwidth; it
takes a few minutes.

## License

MIT — see [LICENSE](LICENSE).

The underlying model is **UVDoc** (Verhoeven et al., SIGGRAPH Asia 2023),
[MIT-licensed](https://github.com/tanguymagne/UVDoc). The bundled weights are
an ONNX export of that model.

## Why this exists

- UVDoc is MIT, but the upstream repo ships a **PyTorch demo only** — no ONNX
  export, and a torch dependency is heavy for a preprocessing step.
- The [`docuwarp`](https://pypi.org/project/docuwarp/) PyPI package wraps the
  same model but **ships no license** (citation only) and is dormant.
- So: export the MIT model to ONNX once, ship the artifact, and run it with
  `onnxruntime` alone. Clean license chain, tiny runtime footprint.
