"""自适应二值化模块（P4）。

reconstruction-assumption: 论文 1.1 节未给出显式二值化判决式 T(x,y)。
本模块提供三种候选策略，全部标记 reconstruction-assumption：

  "mu+S":           T = mu_loc + S
  "mu+C*S":         T = mu_loc + C * S        （默认，S 为论文公式 (5) 的显著性度量）
  "mu+C*sigma_loc": T = mu_loc + C * sigma_loc（对照论文 1.2.1 节
                    T = W_mu - C * W_sigma 的形式，C=1.5）

其中 mu_loc / sigma_loc 为 20x20 块局部统计图（已最近邻放大回原尺寸，
见 local_background.py），S 为同一分块网格上的显著性度量图。

判决规则（engineering-choice）：image >= T 的像素置 255（前景），否则 0，
输出 uint8 二值图。
"""

import numpy as np

STRATEGIES = ("mu+S", "mu+C*S", "mu+C*sigma_loc")


def threshold_map(local_mean_map, S_map, local_std_map, strategy="mu+C*S", C=1.5):
    """按候选策略构造逐像素阈值图 T。"""
    mu = np.asarray(local_mean_map, dtype=np.float32)
    if strategy == "mu+S":
        t = mu + np.asarray(S_map, dtype=np.float32)
    elif strategy == "mu+C*S":
        t = mu + float(C) * np.asarray(S_map, dtype=np.float32)
    elif strategy == "mu+C*sigma_loc":
        t = mu + float(C) * np.asarray(local_std_map, dtype=np.float32)
    else:
        raise ValueError(
            f"未知二值化策略 {strategy!r}，候选: {', '.join(STRATEGIES)}"
        )
    return t


def threshold_strategy(image, local_mean_map, S_map, local_std_map,
                       strategy="mu+C*S", C=1.5):
    """自适应二值化入口。

    返回 (binary, T)：
      binary: uint8 {0, 255} 二值图，image >= T 为前景。
      T: float32 阈值图（与输入同尺寸）。
    """
    t = threshold_map(local_mean_map, S_map, local_std_map,
                      strategy=strategy, C=C)
    img = np.asarray(image, dtype=np.float32)
    if img.shape != t.shape:
        raise ValueError(f"图像尺寸 {img.shape} 与阈值图尺寸 {t.shape} 不一致")
    binary = np.where(img >= t, 255, 0).astype(np.uint8)
    return binary, t
