"""SNR 评价指标模块。

paper-confirmed: SNR = |m - m_b| / sigma_b，其中 m 为目标区域均值，
m_b / sigma_b 为背景区域均值/标准差（线性比值，论文称 dB）。

reconstruction-assumption: ROI 形状论文未规定，采用圆形目标区 +
同心背景环（inner/outer radius），见 data/roi/roi.yaml。
"""

import numpy as np


def _circular_mask(shape, cx, cy, r_inner, r_outer):
    """生成圆环掩膜（r_inner=0 时为实心圆）。坐标约定：x 为列，y 为行。"""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return (dist2 >= r_inner**2) & (dist2 <= r_outer**2)


def calculate_snr(image, target, background):
    """计算 SNR = |m - m_b| / sigma_b。

    target: dict(x, y, radius)，目标圆盘。
    background: dict(inner_radius, outer_radius)，与目标同心的背景环。
    """
    cx, cy = int(target["x"]), int(target["y"])
    tmask = _circular_mask(image.shape, cx, cy, 0, float(target["radius"]))
    bmask = _circular_mask(
        image.shape, cx, cy,
        float(background["inner_radius"]), float(background["outer_radius"]),
    )
    m = float(image[tmask].mean())
    bvals = image[bmask]
    m_b = float(bvals.mean())
    sigma_b = float(bvals.std())
    if sigma_b <= 0:
        raise ValueError("背景区域标准差为 0，无法计算 SNR")
    snr = abs(m - m_b) / sigma_b
    return {
        "snr": snr,
        "target_mean": m,
        "background_mean": m_b,
        "background_std": sigma_b,
    }
