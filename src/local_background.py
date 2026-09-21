"""局部背景建模模块（P3）。

paper-confirmed: 论文 1.1 节将图像划分为 20x20 子块，统计每块局部均值
mu_loc 与局部标准差 sigma_loc，并以全局标准差 sigma_glo 定义显著性度量
（公式 5）：

    S(x, y) = [1 - exp(-sigma_loc^2 / sigma_glo^2)] * sigma_glo

paper-confirmed: sigma_glo 为整幅图像的全局标准差。

engineering-choice: 块统计图按最近邻方式放大回原图尺寸（逐像素取值），
便于后续 P4 自适应阈值逐像素判决；边缘不足 20 像素的残块按实际像素统计
（4096 % 20 = 16，最后一行/列为 16 像素宽的残块）。

engineering-choice: S_map 由块级 sigma_loc 计算，与 mu_loc/sigma_loc 图
保持同一套分块网格；sigma_loc=0 的平坦块 S=0。
"""

import numpy as np


def block_statistics(image, block_size=20):
    """按 block_size 分块统计局部均值/标准差。

    返回 dict：
      local_mean_map / local_std_map: 与原图同尺寸的 float32 块统计图
        （每块内所有像素取该块统计值）。
      block_grid: (n_rows, n_cols) 块网格形状。
      block_mean / block_std: 块级二维数组（n_rows, n_cols）。
    """
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape
    b = int(block_size)
    if b <= 0:
        raise ValueError("block_size 必须为正整数")

    ys = list(range(0, h, b))
    xs = list(range(0, w, b))
    n_rows, n_cols = len(ys), len(xs)
    block_mean = np.zeros((n_rows, n_cols), dtype=np.float64)
    block_std = np.zeros((n_rows, n_cols), dtype=np.float64)

    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            blk = img[y0:min(y0 + b, h), x0:min(x0 + b, w)]
            block_mean[i, j] = float(blk.mean())
            block_std[i, j] = float(blk.std())

    # 最近邻放大回原图尺寸
    row_idx = np.minimum(np.arange(h) // b, n_rows - 1)
    col_idx = np.minimum(np.arange(w) // b, n_cols - 1)
    local_mean_map = block_mean[row_idx][:, col_idx].astype(np.float32)
    local_std_map = block_std[row_idx][:, col_idx].astype(np.float32)

    return {
        "local_mean_map": local_mean_map,
        "local_std_map": local_std_map,
        "block_grid": (n_rows, n_cols),
        "block_mean": block_mean,
        "block_std": block_std,
    }


def compute_s_map(local_std_map, global_std):
    """按论文公式 (5) 计算显著性度量 S_map。

    paper-confirmed: S(x,y) = [1 - exp(-sigma_loc^2 / sigma_glo^2)] * sigma_glo。
    """
    sigma_loc = np.asarray(local_std_map, dtype=np.float64)
    sg = float(global_std)
    if sg <= 0:
        raise ValueError("全局标准差 sigma_glo 为 0，无法计算 S_map")
    s = (1.0 - np.exp(-(sigma_loc**2) / (sg**2))) * sg
    return s.astype(np.float32)


def local_background_model(image, block_size=20):
    """局部背景建模入口：块统计 + 全局标准差 + S_map。

    返回 dict：local_mean_map, local_std_map, global_std, S_map, block_grid。
    """
    stats = block_statistics(image, block_size)
    global_std = float(np.asarray(image, dtype=np.float64).std())
    s_map = compute_s_map(stats["local_std_map"], global_std)
    return {
        "local_mean_map": stats["local_mean_map"],
        "local_std_map": stats["local_std_map"],
        "global_std": global_std,
        "S_map": s_map,
        "block_grid": stats["block_grid"],
    }
