#!/usr/bin/env python3
"""Export the UVDoc model (tanguymagne/UVDoc, MIT) to ONNX.

Produces a parity-checked ONNX artifact runnable by onnxruntime alone
(no torch at inference time). The bundled ``uvdoc_onnx/models/uvdoc.onnx``
was produced by this script.

Usage:
    python scripts/export_uvdoc_onnx.py [OUTPUT.onnx]

Defaults OUTPUT to the packaged location (uvdoc_onnx/models/uvdoc.onnx).
Requires: git, a writable temp dir, and ~2 GB of bandwidth (torch CPU
wheel). Takes a few minutes.

Parity guarantee: the export is validated against the UPSTREAM PYTORCH
MODEL on the upstream repo's own example images BEFORE the artifact is
accepted. Max grid difference > 1e-3 aborts and leaves any existing
artifact untouched.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

UVDOC_REPO = "https://github.com/tanguymagne/UVDoc.git"
UVDOC_COMMIT = "main"  # pin a commit for reproducibility
PACKAGED_OUT = Path(__file__).resolve().parent.parent / "uvdoc_onnx" / "models" / "uvdoc.onnx"

EXPORT_PY = '''
"""One-time UVDoc -> ONNX export with parity check vs the PyTorch demo."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "uvdoc"))
from model import UVDocnet  # noqa: E402
from utils import IMG_SIZE  # noqa: E402

CKPT = Path(__file__).parent / "uvdoc" / "model" / "best_model.pkl"
EXAMPLES = sorted((Path(__file__).parent / "uvdoc" / "examples").glob("*.jpg"))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("uvdoc.onnx")

model = UVDocnet(num_filter=32, kernel_size=5)
# weights_only=False: torch >= 2.6 defaults to True, which rejects the
# checkpoint's non-tensor bookkeeping. Trusted upstream repo, one-time run.
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

dummy = torch.randn(1, 3, IMG_SIZE[1], IMG_SIZE[0])
with torch.no_grad():
    torch.onnx.export(
        model, (dummy,), str(OUT),
        # torch >= 2.6's exporter natively emits opset 18; forcing 17
        # triggers a lossy version-conversion attempt (Pad has no 18->17
        # adapter) that fails loudly. onnxruntime >= 1.17 supports 18.
        opset_version=18,
        input_names=["image"],
        output_names=["grid2d", "grid3d"],
        dynamic_axes=None,  # fixed input size (488x712) by design
    )
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

# --- parity check: ONNX vs PyTorch on the repo's example images ------------
import cv2  # noqa: E402
import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
worst = 0.0
for ex in EXAMPLES:
    img = cv2.imread(str(ex))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    small = cv2.resize(img, IMG_SIZE).transpose(2, 0, 1)[None, ...].astype(np.float32)
    with torch.no_grad():
        pt_grid = model(torch.from_numpy(small))[0].numpy()
    onnx_grid = sess.run(["grid2d"], {inp_name: small})[0]
    diff = float(np.abs(pt_grid - onnx_grid).max())
    worst = max(worst, diff)
    print(f"  {ex.name}: max grid diff {diff:.2e}")
if worst > 1e-3:
    sys.exit(f"PARITY FAILED: max grid diff {worst:.2e} > 1e-3 — keeping old artifact")
print(f"parity OK (worst {worst:.2e})")
'''


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else PACKAGED_OUT
    if not shutil.which("git"):
        print("error: git is required (clones the upstream UVDoc repo)", file=sys.stderr)
        return 1

    work = Path(tempfile.mkdtemp(prefix="uvdoc-export-"))
    try:
        print(f"==> Cloning UVDoc ({UVDOC_COMMIT})…")
        subprocess.run(["git", "clone", "--quiet", UVDOC_REPO, str(work / "uvdoc")],
                       check=True)
        subprocess.run(["git", "-C", str(work / "uvdoc"), "checkout", "--quiet",
                        UVDOC_COMMIT], check=True)

        print("==> Building throwaway venv (torch CPU)…")
        venv_py = work / "venv" / "bin" / "python"
        subprocess.run([sys.executable, "-m", "venv", str(work / "venv")], check=True)
        pip = [str(venv_py), "-m", "pip", "install", "--quiet"]
        subprocess.run(pip + ["--upgrade", "pip"], check=True)
        subprocess.run(pip + ["torch", "--index-url",
                              "https://download.pytorch.org/whl/cpu"], check=True)
        # onnxscript: required by torch >= 2.6's ONNX export machinery.
        subprocess.run(pip + ["numpy", "opencv-python-headless",
                              "onnxruntime", "onnxscript"], check=True)

        (work / "export.py").write_text(EXPORT_PY, encoding="utf-8")
        print("==> Exporting + parity-checking…")
        tmp_out = out.with_suffix(out.suffix + ".tmp")
        subprocess.run([str(venv_py), str(work / "export.py"), str(tmp_out)],
                       check=True, cwd=work)

        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            bak = out.with_suffix(out.suffix + ".bak")
            out.replace(bak)
            print(f"(previous artifact kept at {bak})")
        tmp_out.replace(out)
        print(f"==> Done: {out}")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
