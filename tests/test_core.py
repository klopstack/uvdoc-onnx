"""uvdoc_onnx.core: UVDoc ONNX wrapper — numerics + failure contract.

onnxruntime is NOT required here: the session is injected via the module's
cache (``core._sessions``), and the failure path is exercised with a
nonexistent model file (works with or without onnxruntime installed).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from uvdoc_onnx import DEFAULT_MODEL_PATH, DewarpError, dewarp  # noqa: E402
from uvdoc_onnx import core  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_sessions():
    saved = dict(core._sessions)
    yield
    core._sessions.clear()
    core._sessions.update(saved)


class _FakeSession:
    """Stands in for an ORT InferenceSession: returns a canned grid."""

    def __init__(self, grid):
        self._grid = grid  # (2, Gh, Gw)

    def get_inputs(self):
        return [SimpleNamespace(name="image")]

    def run(self, output_names, inputs):
        return [self._grid[None, ...]]  # add batch dim


def _identity_grid(gh=45, gw=31):
    """Grid that maps every output point to itself (the no-op transform).

    Defaults match the real UVDoc output: (2, Gh=45, Gw=31).
    """
    cols = (-1.0 + 2.0 * np.arange(gw) / (gw - 1)).reshape(1, 1, gw)
    rows = (-1.0 + 2.0 * np.arange(gh) / (gh - 1)).reshape(1, gh, 1)
    gx = np.broadcast_to(cols, (1, gh, gw))
    gy = np.broadcast_to(rows, (1, gh, gw))
    return np.concatenate([gx, gy], axis=0).astype(np.float32)


# --- packaging ----------------------------------------------------------------

def test_bundled_weights_ship():
    assert DEFAULT_MODEL_PATH.is_file(), "bundled uvdoc.onnx missing from package"
    assert DEFAULT_MODEL_PATH.stat().st_size > 100_000  # ~214 KB, not a stub


# --- _upsample_grid (F.interpolate align_corners=True replica) -------------

def test_upsample_constant_grid_stays_constant():
    grid = np.zeros((2, 45, 31), dtype=np.float32)
    out = core._upsample_grid(grid, 100, 200)
    assert out.shape == (2, 100, 200)
    assert np.allclose(out, 0.0, atol=1e-6)


def test_upsample_linear_ramp_is_exact():
    # align_corners=True: a linear field is reproduced exactly at any size.
    grid = _identity_grid(45, 31)
    out = core._upsample_grid(grid, 712, 488)
    exp_x = np.broadcast_to(
        (-1.0 + 2.0 * np.arange(488) / 487).reshape(1, 1, 488), (1, 712, 488))
    exp_y = np.broadcast_to(
        (-1.0 + 2.0 * np.arange(712) / 711).reshape(1, 712, 1), (1, 712, 488))
    assert np.allclose(out[0], exp_x, atol=1e-4)
    assert np.allclose(out[1], exp_y, atol=1e-4)


def test_upsample_midpoint_of_unit_square():
    # 2x2 grid spanning [-1, 1]; the centre of a 3x3 upscale must be 0.
    grid = np.array([[[[-1.0, 1.0], [-1.0, 1.0]],
                      [[-1.0, -1.0], [1.0, 1.0]]]], dtype=np.float32)
    out = core._upsample_grid(grid[0], 3, 3)
    assert out.shape == (2, 3, 3)
    assert abs(out[0, 1, 1]) < 1e-6  # x-channel centre
    assert abs(out[1, 1, 1]) < 1e-6  # y-channel centre
    assert abs(out[0, 0, 0] + 1.0) < 1e-6  # corner preserved


# --- _grid_sample (F.grid_sample align_corners=True, zero pad) --------------

def test_grid_sample_identity_roundtrip():
    rng = np.random.default_rng(7)
    img = rng.random((64, 96, 3), dtype=np.float32)
    h, w = img.shape[:2]
    cols = (-1.0 + 2.0 * np.arange(w) / (w - 1)).reshape(1, 1, w)
    rows = (-1.0 + 2.0 * np.arange(h) / (h - 1)).reshape(1, h, 1)
    grid = np.concatenate([
        np.broadcast_to(cols, (1, h, w)),
        np.broadcast_to(rows, (1, h, w)),
    ], axis=0).astype(np.float32)
    out = core._grid_sample(img, grid)
    assert np.max(np.abs(out - img)) < 1e-4
    # Regression: the boundary-inclusive check must preserve the LAST
    # row/column (an exclusive check zeros them — a black stripe on the
    # right/bottom edge of every unwarped page).
    assert np.max(np.abs(out[h - 1] - img[h - 1])) < 1e-4
    assert np.max(np.abs(out[:, w - 1] - img[:, w - 1])) < 1e-4


def test_grid_sample_out_of_bounds_is_zero():
    img = np.ones((32, 32, 3), dtype=np.float32)
    grid = np.zeros((2, 32, 32), dtype=np.float32)
    grid[0, :, :] = 1.5  # x shifted far past the right edge
    out = core._grid_sample(img, grid)
    assert np.all(out == 0.0)


# --- dewarp() end-to-end with an injected session ----------------------------

def test_dewarp_identity_grid_preserves_image():
    # Structured image so a broken sampler would be obvious.
    arr = np.zeros((712, 488, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 488, dtype=np.uint8)[None, :]
    img = Image.fromarray(arr, mode="RGB")
    core._sessions["fake"] = _FakeSession(_identity_grid())
    out, changed, err = dewarp(img, "fake")
    assert err is None and changed is True
    assert out.size == img.size
    diff = np.abs(np.asarray(out, dtype=np.int16)
                  - np.asarray(img, dtype=np.int16))
    assert diff.max() <= 2, f"identity dewarp drifted: max diff {diff.max()}"


def test_dewarp_bad_grid_shape_reports_error():
    img = Image.new("RGB", (488, 712), (10, 20, 30))
    core._sessions["bad"] = _FakeSession(np.zeros((2, 4, 4), np.float32))
    out, changed, err = dewarp(img, "bad")
    assert changed is False and err is not None
    assert "grid shape" in err
    assert out is img  # input returned UNCHANGED


def test_dewarp_missing_model_returns_input_and_error(tmp_path):
    img = Image.new("RGB", (300, 400), (1, 2, 3))
    out, changed, err = dewarp(img, str(tmp_path / "nope.onnx"))
    assert changed is False
    assert err  # some error string (missing file or missing onnxruntime)
    assert out is img


def test_dewarp_session_exception_returns_input():
    class _Boom:
        def get_inputs(self):
            return [SimpleNamespace(name="image")]

        def run(self, *_a, **_k):
            raise RuntimeError("cuda went away")

    img = Image.new("RGB", (488, 712), (9, 9, 9))
    core._sessions["boom"] = _Boom()
    out, changed, err = dewarp(img, "boom")
    assert changed is False and "cuda went away" in err
    assert out is img


def test_get_session_real_path_loads_and_caches(tmp_path, monkeypatch):
    """Exercise _get_session WITHOUT cache injection.

    Cache-injecting tests bypass this function entirely — which is how a
    missing module-level logger (referenced by its session-load log line)
    survived to production. A fake onnxruntime module stands in so the
    real code path (file check → SessionOptions → InferenceSession →
    log → cache) runs.
    """
    import types
    fake_ort = types.ModuleType("onnxruntime")

    class _Opts:
        inter_op_num_threads = 0
        intra_op_num_threads = 0

    class _Sess:
        def __init__(self, path, sess_options=None, providers=None):
            self.path = path

    fake_ort.SessionOptions = _Opts
    fake_ort.InferenceSession = _Sess
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    model = tmp_path / "uvdoc.onnx"
    model.write_bytes(b"fake-onnx")
    sess = core._get_session(str(model))
    assert sess.path == str(model)
    assert core._sessions[str(model)] is sess  # cached for next call


def test_default_model_path_points_at_bundled_weights():
    assert str(DEFAULT_MODEL_PATH).endswith("uvdoc.onnx")
    assert DEFAULT_MODEL_PATH.is_file()
