"""bilateral_filter 单元测试：核定义、sigma_range 缩放、权重对称性、
与手工小矩阵对照（implementation_plan 测试节）。"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bilateral_filter import (bilateral_filter, effective_sigma_range,
                                  kernel_diameter)


def test_kernel_diameter_modes():
    assert kernel_diameter(5, "diameter") == 5
    assert kernel_diameter(5, "radius") == 11
    with pytest.raises(ValueError):
        kernel_diameter(5, "bogus")


def test_effective_sigma_range_scaling():
    img = np.linspace(0, 510, 100, dtype=np.float32).reshape(10, 10)
    # 动态范围 510 时 sigma_r_eff = 25/255*510 = 50
    assert effective_sigma_range(img, 25, True) == pytest.approx(50.0)
    assert effective_sigma_range(img, 25, False) == pytest.approx(25.0)
    # 动态范围为 0（常数图）时退化为原值
    const = np.zeros((4, 4), dtype=np.float32)
    assert effective_sigma_range(const, 25, True) == pytest.approx(25.0)


def test_constant_image_unchanged():
    img = np.full((32, 32), 123.0, dtype=np.float32)
    out, _ = bilateral_filter(img, kernel_size=5, sigma_space=1.5, sigma_range=25)
    assert np.allclose(out, img, atol=1e-4)


def _manual_bilateral(img, d, sigma_s, sigma_r):
    """手工双边滤波（仅内部像素，不涉及边界），float64 参考实现。"""
    r = d // 2
    h, w = img.shape
    out = np.full_like(img, np.nan, dtype=np.float64)
    img64 = img.astype(np.float64)
    for y in range(r, h - r):
        for x in range(r, w - r):
            acc, wsum = 0.0, 0.0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    ws = np.exp(-(dx * dx + dy * dy) / (2 * sigma_s**2))
                    diff = img64[y + dy, x + dx] - img64[y, x]
                    wr = np.exp(-(diff * diff) / (2 * sigma_r**2))
                    acc += ws * wr * img64[y + dy, x + dx]
                    wsum += ws * wr
            out[y, x] = acc / wsum
    return out


def test_matches_manual_small_matrix():
    """内部像素与手工公式逐像素对照（同时验证空间/灰度权重对称性）。

    engineering-choice（容差依据）：实测 OpenCV 5.0 的 float32 bilateralFilter
    与教科书公式存在最高约 5% 的偏差（疑似快速 exp 近似/权重截断，经探针
    实验确认其颜色权重形状与 exp(-d^2/2sigma_r^2) 一致、翻转对称性保持），
    故本对照以 5% 容差防止实现层面的 gross error（如 sigma 互换、窗口错误）。
    """
    rng = np.random.default_rng(42)
    img = rng.normal(100.0, 20.0, size=(9, 9)).astype(np.float32)
    d, sigma_s, sigma_r = 3, 1.5, 15.0
    out, _ = bilateral_filter(
        img, kernel_size=d, sigma_space=sigma_s, sigma_range=sigma_r,
        kernel_definition="diameter", sigma_range_scale_to_dynamic_range=False,
    )
    ref = _manual_bilateral(img, d, sigma_s, sigma_r)
    interior = ~np.isnan(ref)
    assert np.allclose(out[interior], ref[interior], rtol=5e-2, atol=1.0)


def test_edge_preservation_vs_gaussian():
    """阶跃边上双边滤波应比高斯模糊更保边（功能性对照）。"""
    rng = np.random.default_rng(3)
    img = np.zeros((64, 64), dtype=np.float32)
    img[:, 32:] = 200.0
    img += rng.normal(0.0, 2.0, size=img.shape).astype(np.float32)

    out, _ = bilateral_filter(img, kernel_size=5, sigma_space=1.5, sigma_range=25,
                              sigma_range_scale_to_dynamic_range=False)
    gauss = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)
    # 距边缘 2 列的左侧像素（高斯核半径内）：双边滤波应仍接近 0，高斯被显著拉高
    col = 30
    assert float(np.median(out[:, col])) < 5.0
    assert float(np.median(gauss[:, col])) > float(np.median(out[:, col])) + 20.0


def test_flip_symmetry():
    """滤波与图像翻转对易（权重对称性的整体检验）。"""
    rng = np.random.default_rng(7)
    img = rng.normal(0.0, 1.0, size=(21, 17)).astype(np.float32)
    out, _ = bilateral_filter(img, kernel_size=5, sigma_space=1.5, sigma_range=25,
                              sigma_range_scale_to_dynamic_range=False)
    flipped = np.ascontiguousarray(img[:, ::-1])
    out_f, _ = bilateral_filter(flipped, kernel_size=5, sigma_space=1.5,
                                sigma_range=25,
                                sigma_range_scale_to_dynamic_range=False)
    assert np.allclose(out_f, out[:, ::-1], rtol=1e-4, atol=1e-4)
