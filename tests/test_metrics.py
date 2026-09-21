"""metrics 单元测试：合成图像 SNR 已知值、detection_metrics 合成场景
（implementation_plan 测试节）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import calculate_snr, detection_metrics


def _annulus_mask(shape, cx, cy, r_in, r_out):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return (d2 >= r_in**2) & (d2 <= r_out**2)


def test_snr_known_value():
    """背景环一半 95 一半 105（mean=100, std=5），目标盘 130：
    SNR = |130 - 100| / 5 = 6。"""
    shape = (61, 61)
    cx, cy = 30, 30
    img = np.full(shape, 100.0, dtype=np.float32)
    bmask = _annulus_mask(shape, cx, cy, 10, 20)
    vals = np.full(int(bmask.sum()), 95.0, dtype=np.float32)
    vals[::2] = 105.0     # 偶数个时恰好各半；奇数差一个，用 approx 容忍
    img[bmask] = vals
    # 目标盘 r=5 填 130（盘在环内径 10 以内，不污染背景环）
    tmask = _annulus_mask(shape, cx, cy, 0, 5)
    img[tmask] = 130.0

    s = calculate_snr(img, {"x": cx, "y": cy, "radius": 5},
                      {"inner_radius": 10, "outer_radius": 20})
    assert s["target_mean"] == pytest.approx(130.0)
    assert s["background_mean"] == pytest.approx(100.0, abs=0.2)
    assert s["background_std"] == pytest.approx(5.0, abs=0.2)
    assert s["snr"] == pytest.approx(6.0, abs=0.2)


def test_snr_zero_background_std_raises():
    img = np.full((31, 31), 7.0, dtype=np.float32)
    with pytest.raises(ValueError):
        calculate_snr(img, {"x": 15, "y": 15, "radius": 3},
                      {"inner_radius": 6, "outer_radius": 10})


def test_detection_metrics_synthetic():
    """合成：目标盘内全部前景（能量全保留）+ 远处一个孤立噪点连通域。"""
    shape = (61, 61)
    cx, cy = 30, 30
    gray = np.full(shape, 100.0, dtype=np.float32)
    tmask = _annulus_mask(shape, cx, cy, 0, 5)
    gray[tmask] = 200.0                       # 目标盘能量 200/px

    binary = np.zeros(shape, dtype=np.uint8)
    binary[tmask] = 255                       # 目标盘全保留
    binary[5, 5] = 255                        # 孤立噪点（与目标盘不相交）

    det = detection_metrics(binary, gray, {"x": cx, "y": cy, "radius": 5})
    assert det["target_energy_retention"] == pytest.approx(1.0)
    assert det["target_pixel_retention"] == pytest.approx(1.0)
    assert det["background_noise_count"] == 1
    assert det["foreground_pixels"] == int(tmask.sum()) + 1


def test_detection_metrics_partial_retention():
    """目标盘只保留一半像素时能量保留率约 0.5，无噪点。"""
    shape = (41, 41)
    cx, cy = 20, 20
    gray = np.zeros(shape, dtype=np.float32)
    tmask = _annulus_mask(shape, cx, cy, 0, 4)
    gray[tmask] = 100.0
    binary = np.zeros(shape, dtype=np.uint8)
    half = tmask.copy()
    half[:, cx:] = False                      # 只保留左半盘
    binary[half] = 255

    det = detection_metrics(binary, gray, {"x": cx, "y": cy, "radius": 4})
    total = int(tmask.sum())
    kept = int(half.sum())
    assert det["target_energy_retention"] == pytest.approx(kept / total)
    assert det["background_noise_count"] == 0
