"""Portability checks for loading a calibration table only when requested."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cowler.io import lut as lut_module
from cowler.io.normalize import lut_params, normalize_signal


def test_missing_laboratory_table_does_not_block_signal_normalisation(monkeypatch, tmp_path):
    monkeypatch.setenv("COWLER_LUT_PATH", str(tmp_path / "absent.csv"))
    monkeypatch.setattr(lut_module, "LUT_6mer", None)
    np.testing.assert_allclose(normalize_signal(np.array([1., 2., 3.])), [-1/1.4826, 0, 1/1.4826])
    with pytest.raises(FileNotFoundError, match="COWLER_LUT_PATH"):
        lut_module.predict_DNA_6mer_5_3("AAAAAAAA")


def test_external_table_is_used_by_prediction_and_normalisation(monkeypatch, tmp_path: Path):
    table = pd.DataFrame({
        "kmer_pull_3_5": ["AAAAAA"], "pre_mean": [1.0], "pre_std": [0.1],
        "post_mean": [3.0], "post_std": [0.2],
    })
    path = tmp_path / "calibration.csv"
    table.to_csv(path, index=False)
    monkeypatch.setenv("COWLER_LUT_PATH", str(path))
    monkeypatch.setattr(lut_module, "LUT_6mer", None)
    prediction = lut_module.predict_DNA_6mer_5_3("AAAAAAAA")
    np.testing.assert_allclose(prediction["mean"].dropna(), [1, 3, 1, 3, 1, 3])
    assert lut_params() == pytest.approx((2.0, 1.4826))
    pd.testing.assert_frame_equal(lut_module.get_default_lut(), table)
