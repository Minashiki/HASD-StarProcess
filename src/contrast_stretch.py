"""对比度拉伸模块（P3）。

paper-confirmed: 论文 1.1 节采用 min-max 线性拉伸到 [0,255]，
不使用 CLAHE / 百分位截断等非线性方法。

engineering-choice: 拉伸作用于双边滤波之后的 float32 数据（论文流程顺序：
滤波 → 拉伸 → 局部背景建模）。rst19 数据存在 ±32768 量级的坏点，全局 min-max
会被坏点主导（星点被压缩到低位灰度），这与论文隐含假设一致、如实实现，
其影响由后续局部背景建模与二值化缓解；显示用 PNG 的百分位拉伸仅影响落盘图，
不影响管线数据。
"""

import numpy as np


def contrast_stretch(image, out_min=0.0, out_max=255.0, method="minmax"):
    """min-max 线性拉伸到 [out_min, out_max]，返回 float32。

    paper-confirmed: method 固定为 minmax（论文方法）。
    """
    if method != "minmax":
        raise ValueError(f"未知拉伸方法: {method}（paper-confirmed 仅支持 minmax）")
    img = np.asarray(image, dtype=np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.full(img.shape, float(out_min), dtype=np.float32)
    out = (img - lo) / (hi - lo) * (float(out_max) - float(out_min)) + float(out_min)
    return out.astype(np.float32)
