"""local_background 单元测试：块统计正确性、边缘残块、S_map 公式
（implementation_plan 测试节）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.local_background import (block_statistics, compute_s_map,
                                  local_background_model)


def test_block_statistics_known_values():
    """40x40 图、20x20 块：四块各填已知常数/噪声，核对均值与标准差。"""
    img = np.zeros((40, 40), dtype=np.float32)
    img[0:20, 0:20] = 10.0                      # 常数块：mean=10, std=0
    blk = np.arange(400, dtype=np.float32).reshape(20, 20)
    img[0:20, 20:40] = blk                      # 已知块
    img[20:40, 0:20] = 5.0
    img[20:40, 20:40] = 7.5

    st = block_statistics(img, block_size=20)
    assert st["block_grid"] == (2, 2)
    bm, bs = st["block_mean"], st["block_std"]
    assert bm[0, 0] == pytest.approx(10.0)
    assert bs[0, 0] == pytest.approx(0.0)
    assert bm[0, 1] == pytest.approx(float(blk.mean()))
    assert bs[0, 1] == pytest.approx(float(blk.std()))
    assert bm[1, 0] == pytest.approx(5.0)
    assert bm[1, 1] == pytest.approx(7.5)

    # 最近邻放大回原尺寸：块内所有像素取块统计值
    assert st["local_mean_map"].shape == (40, 40)
    assert np.all(st["local_mean_map"][0:20, 0:20] == pytest.approx(10.0))
    assert np.all(st["local_std_map"][20:40, 20:40] == pytest.approx(0.0))


def test_edge_remainder_blocks():
    """45x45 图、20x20 块：3x3 网格，末行/末列为 5 像素宽残块。"""
    img = np.zeros((45, 45), dtype=np.float32)
    img[40:45, 40:45] = 3.0     # 5x5 残块
    st = block_statistics(img, block_size=20)
    assert st["block_grid"] == (3, 3)
    assert st["block_mean"][2, 2] == pytest.approx(3.0)
    assert st["block_std"][2, 2] == pytest.approx(0.0)
    # 残块统计值映射回原图对应区域
    assert np.all(st["local_mean_map"][40:45, 40:45] == pytest.approx(3.0))


def test_compute_s_map_formula():
    """paper-confirmed 公式 (5)：S = [1 - exp(-sigma_loc^2/sigma_glo^2)] * sigma_glo。"""
    sigma_loc = np.array([[0.0, 2.0]], dtype=np.float32)
    sg = 2.0
    s = compute_s_map(sigma_loc, sg)
    assert s[0, 0] == pytest.approx(0.0)                      # 平坦块 S=0
    assert s[0, 1] == pytest.approx((1.0 - np.exp(-1.0)) * 2.0)
    with pytest.raises(ValueError):
        compute_s_map(sigma_loc, 0.0)


def test_local_background_model_global_std():
    rng = np.random.default_rng(1)
    img = rng.normal(50.0, 3.0, size=(60, 60)).astype(np.float32)
    bg = local_background_model(img, block_size=20)
    assert bg["block_grid"] == (3, 3)
    assert bg["global_std"] == pytest.approx(float(img.astype(np.float64).std()))
    assert bg["S_map"].shape == img.shape
    assert np.all(bg["S_map"] >= 0.0)
