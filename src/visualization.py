"""中间结果可视化/落盘模块。

engineering-choice: PNG 仅用于人工检查，保存前做 min-max 归一化到 uint8；
显示归一化不影响管线中的 float32 数据。
"""

from pathlib import Path

import cv2
import numpy as np


def normalize_to_uint8(image, lo_pct=0.5, hi_pct=99.5):
    """百分位拉伸到 [0,255] uint8（仅用于图像落盘显示）。

    engineering-choice: rst19 数据存在 0/65535 量级的坏点（uint16 饱和/零值），
    全局 min-max 会把星点压缩到不可见，故显示归一化用 0.5%/99.5% 百分位；
    管线内的对比度拉伸（P3）仍按论文做 min-max，两者互不影响。
    """
    lo, hi = np.percentile(image, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    return ((image - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)


def save_png(path, image):
    """保存灰度图为 PNG（自动建目录）。返回路径字符串。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8:
        image = normalize_to_uint8(np.asarray(image, dtype=np.float32))
    cv2.imwrite(str(path), image)
    return str(path)


def overlay_mask(image, binary, color=(0, 0, 255)):
    """在灰度图的显示归一化结果上，用 color（BGR）标出前景像素。

    engineering-choice: 仅用于 09_final.png 人工检查；返回 BGR uint8。
    """
    base = normalize_to_uint8(np.asarray(image, dtype=np.float32))
    bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    bgr[np.asarray(binary) > 0] = color
    return bgr
