"""Two-step LUT (pore model) loader and half-step profile builder.

LUT columns per 6-mer: kmer_pull_3_5, pre_mean, pre_std, post_mean, post_std (pA).
k = 6 => 4096 k-mers x 4 values. If stds missing, fit from Component 1 residuals
before trusting likelihoods (Component 2).

Half-step profile for a template (length N), M = N - k + 1 k-mers, L = 2M:
    E_mean[2j]   = pre_mean[j]    E_std[2j]   = pre_std[j]
    E_mean[2j+1] = post_mean[j]   E_std[2j+1] = post_std[j]
Half-step h -> k-mer h // 2; reported base = (h // 2) + BASE_OFFSET.
predict_DNA_6mer_5_3 uses a constriction offset of 4 (verify against a control).

Public API:
    load_lut(csv_filename)             -- load an externally supplied LUT table
    predict_DNA_6mer_5_3(template, lut)-- interleaved pre/post profile DataFrame
"""

from __future__ import annotations

import os
from pathlib import Path
from importlib import resources as impresources
from importlib.resources.abc import Traversable

import numpy as np
import numpy.typing as npt
import pandas as pd


def get_model_path(csv_filename: str = "DNA_6mer_prediction_model_cv.csv") -> Traversable:
    """Locate an external calibration CSV, with a legacy assets-path fallback."""
    override = os.environ.get("COWLER_LUT_PATH")
    if override:
        return Path(override).expanduser()
    return impresources.files("cowler.io") / "assets" / csv_filename


def load_lut(csv_filename: str = "DNA_6mer_prediction_model_cv.csv") -> pd.DataFrame:
    """Read the LUT into a DataFrame via a real filesystem path (typed-safe)."""
    resource = get_model_path(csv_filename)
    if not resource.is_file():
        raise FileNotFoundError(
            "The laboratory DNA calibration table is not included. "
            "Set COWLER_LUT_PATH to an authorised calibration CSV, "
            "or pass a DataFrame via the lut argument. "
            "See docs/experimental-runners.md."
        )
    with impresources.as_file(resource) as path:
        return pd.read_csv(path, encoding="utf-8")


# The portfolio ships no laboratory calibration data. Load only on demand.
LUT_6mer: pd.DataFrame | None = None


def get_default_lut() -> pd.DataFrame:
    """Lazily load and cache the user-supplied DNA calibration table."""
    global LUT_6mer
    if LUT_6mer is None:
        LUT_6mer = load_lut()
    return LUT_6mer


def moving_6mer_substrings(string: str, window_size: int = 6) -> list[str]:
    return [string[i:i + window_size] for i in range(len(string) - window_size + 1)]


def predict_DNA_6mer_5_3(template: str, lut: pd.DataFrame | None = None) -> pd.DataFrame:
    lut = get_default_lut() if lut is None else lut
    template = template.upper()[::-1]  # uppercase, then reverse to 3'->5'
    constriction_kmer_offset = -4

    sub_6mer = moving_6mer_substrings(template)

    # Vectorized LUT lookup: reindex fills NaN for any k-mer missing from the LUT.
    lut_by_kmer = lut.set_index('kmer_pull_3_5')
    idx = pd.Index(sub_6mer)

    def column(name: str) -> npt.NDArray[np.float64]:
        return lut_by_kmer[name].reindex(idx).to_numpy(dtype=np.float64)

    # Interleave pre/post: [pre_0, post_0, pre_1, post_1, ...]
    DNA_prediction_mean: npt.NDArray[np.float64] = np.empty(len(sub_6mer) * 2)
    DNA_prediction_std: npt.NDArray[np.float64] = np.empty(len(sub_6mer) * 2)
    DNA_prediction_mean[0::2] = column('pre_mean')
    DNA_prediction_mean[1::2] = column('post_mean')
    DNA_prediction_std[0::2] = column('pre_std')
    DNA_prediction_std[1::2] = column('post_std')

    # Organize results into DataFrame
    result_length = 2 * len(template)
    mean: npt.NDArray[np.float64] = np.full(result_length, np.nan)
    std: npt.NDArray[np.float64] = np.full(result_length, np.nan)
    base: list[str] = [''] * result_length
    mode: list[str] = [''] * result_length
    step: npt.NDArray[np.int_] = np.arange(0, result_length) + constriction_kmer_offset

    start_index = -constriction_kmer_offset
    end_index = len(DNA_prediction_mean) - constriction_kmer_offset
    mean[start_index:end_index] = DNA_prediction_mean
    std[start_index:end_index] = DNA_prediction_std

    for str_i in range(len(template)):
        base[2 * str_i] = template[str_i]
        base[2 * str_i + 1] = template[str_i]
        mode[2 * str_i] = 'pre'
        mode[2 * str_i + 1] = 'post'

    return pd.DataFrame({
        'step': step,
        'base': base,
        'mode': mode,
        'mean': mean,
        'std': std,
    })
