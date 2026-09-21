"""contrast_stretch 单元测试：MAD 背景估计、statistical 上限与截断归一化。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contrast_stretch import (
    contrast_stretch,
    estimate_background_mad,
    statistical_upper,
)


def _gaussian_background(shape=(101, 101), mean=100.0, sigma=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, sigma, shape).astype(np.float32)


def test_mad_estimates_gaussian_sigma():
    img = _gaussian_background(mean=100.0, sigma=3.0)
    median, sigma = estimate_background_mad(img)
    assert median == pytest.approx(100.0, abs=0.5)
    assert sigma == pytest.approx(3.0, abs=0.3)


def test_mad_ignores_nan_inf():
    img = _gaussian_background()
    img[0, 0] = np.nan
    img[0, 1] = np.inf
    median, sigma = estimate_background_mad(img)
    assert median == pytest.approx(100.0, abs=0.5)
    assert sigma == pytest.approx(3.0, abs=0.3)


def test_statistical_upper_uses_percentile_when_outliers():
    """大量离群坏点时 percentile > median + k*sigma，upper 取百分位值。"""
    img = _gaussian_background()
    img[:5, :] = 65535.0                       # 约 5% 饱和坏点
    tiers = {"recall": 99.5, "balanced": 99.9, "purity": 99.95}
    upper, median, sigma = statistical_upper(img, "balanced", tiers, upper_k=8.0)
    expected_pct = float(np.percentile(img, 99.9))
    assert upper == pytest.approx(max(expected_pct, median + 8.0 * sigma))


def test_statistical_upper_floor_by_background():
    """无离群点时 MAD 下限项更大，upper = median + 8*sigma。"""
    img = _gaussian_background(mean=100.0, sigma=3.0)
    upper, median, sigma = statistical_upper(img, "balanced", upper_k=8.0)
    assert upper == pytest.approx(median + 8.0 * sigma, rel=1e-6)


def test_statistical_stretch_clips_and_normalizes():
    """大于 upper 的像素被截断到 upper，输出范围 [0, 255]。"""
    img = _gaussian_background(mean=100.0, sigma=3.0)
    img[0, 0] = 65535.0
    out = contrast_stretch(img, method="statistical", tier="balanced")
    assert out.dtype == np.float32
    assert out.max() == pytest.approx(255.0)
    assert out.min() >= 0.0
    upper, _, _ = statistical_upper(img, "balanced")
    lo = float(img.min())
    expected = (upper - lo) / (upper - lo) * 255.0
    assert out[0, 0] == pytest.approx(expected)


def test_statistical_unknown_tier_raises():
    img = _gaussian_background()
    with pytest.raises(ValueError):
        contrast_stretch(img, method="statistical", tier="nonexistent")


def test_minmax_unchanged():
    img = np.array([[10.0, 20.0], [30.0, 60.0]], dtype=np.float32)
    out = contrast_stretch(img, method="minmax")
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(255.0)
    assert out[0, 1] == pytest.approx((20.0 - 10.0) / 50.0 * 255.0)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        contrast_stretch(np.zeros((3, 3), dtype=np.float32), method="clahe")
