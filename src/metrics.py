"""SNR 评价指标模块。

paper-confirmed: SNR = |m - m_b| / sigma_b，其中 m 为目标区域均值，
m_b / sigma_b 为背景区域均值/标准差（线性比值，论文称 dB）。

reconstruction-assumption: ROI 形状论文未规定，采用圆形目标区 +
同心背景环（inner/outer radius），见 data/roi/roi.yaml。

P4 追加 detection_metrics：星点能量保留率 + 背景噪点数
（engineering-choice，论文未定义，见函数 docstring）。
"""

import cv2
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


def detection_metrics(binary, image, target):
    """二值/形态学结果评价（P4 选参依据）。

    engineering-choice: implementation_plan P4 规定按"星点能量保留率 +
    背景噪点数"选参，论文未定义具体形式，此处取：

      target_energy_retention: 目标圆盘内被前景保留的灰度能量占
        该圆盘原图总能量（像素灰度和）的比例，越接近 1 说明星点保留越好。
      target_pixel_retention: 目标圆盘内前景像素数占比（辅助观察）。
      background_noise_count: 全图前景连通域（8 连通）中，与目标圆盘
        不相交的连通域个数，即"背景噪点数"。
      foreground_pixels: 全图前景像素总数。

    binary: uint8 {0,255} 二值图（或形态学结果）。
    image: 二值化之前的灰度图（对比度拉伸结果），用于能量统计。
    target: dict(x, y, radius)，SNR 参考目标圆盘。
    """
    fg = np.asarray(binary) > 0
    img = np.asarray(image, dtype=np.float64)
    cx, cy = int(target["x"]), int(target["y"])
    tmask = _circular_mask(fg.shape, cx, cy, 0, float(target["radius"]))

    total_energy = float(img[tmask].sum())
    kept_energy = float(img[tmask & fg].sum())
    retention = kept_energy / total_energy if total_energy > 0 else 0.0
    pixel_retention = float(fg[tmask].mean()) if tmask.any() else 0.0

    n_labels, labels = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    target_label = 0
    if fg[tmask].any():
        # 目标圆盘内前景像素数最多的连通域视为目标，其余为噪点
        lbls = labels[tmask & fg]
        target_label = int(np.bincount(lbls.ravel()).argmax())
    noise_count = sum(1 for lab in range(1, n_labels) if lab != target_label)

    return {
        "target_energy_retention": retention,
        "target_pixel_retention": pixel_retention,
        "background_noise_count": noise_count,
        "foreground_pixels": int(fg.sum()),
    }
