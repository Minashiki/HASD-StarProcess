"""自适应双边滤波模块。

paper-confirmed: 论文公式 (1)-(4) 为双边滤波（空间高斯 × 灰度高斯），
参数 k=5, sigma_space=1.5, sigma_range=25；核定义支持 diameter 与 radius 两种。
本实现用 OpenCV cv2.bilateralFilter（与论文公式等价）。

engineering-choice: 论文 sigma_range=25 隐含 [0,255] 灰度尺度，但滤波作用于
拉伸前的原始 float32 数据（动态范围远大于 255），故默认按帧动态范围等比缩放
sigma_range：sigma_r_eff = sigma_range / 255 * (max - min)。可通过
sigma_range_scale_to_dynamic_range 关闭（此时 sigma_range 直接按原始灰度解释）。

engineering-choice: 灰度尺度按每帧自身动态范围计算；批处理时各帧独立缩放，
对 15 帧序列的动态范围差异具有自适应性。
"""

import cv2
import numpy as np


def effective_sigma_range(image, sigma_range, scale_to_dynamic_range=True):
    """计算作用于当前图像的等效 sigma_range。

    paper-confirmed: sigma_range=25 为 [0,255] 尺度值。
    engineering-choice: 缩放方式为按帧动态范围线性等比。
    """
    if not scale_to_dynamic_range:
        return float(sigma_range)
    dynamic_range = float(image.max()) - float(image.min())
    if dynamic_range <= 0:
        return float(sigma_range)
    return float(sigma_range) / 255.0 * dynamic_range


def kernel_diameter(kernel_size, kernel_definition="diameter"):
    """将核尺寸换算为 cv2.bilateralFilter 的直径参数 d。

    paper-confirmed: 论文 k=5 存在 diameter（5x5）与 radius（11x11）两种解释，
    主实验用 diameter，radius 作对照（implementation_plan P1/P2）。
    """
    if kernel_definition == "diameter":
        return int(kernel_size)
    if kernel_definition == "radius":
        return 2 * int(kernel_size) + 1
    raise ValueError(f"未知 kernel_definition: {kernel_definition}（应为 diameter|radius）")


def bilateral_filter(image, kernel_size=5, sigma_space=1.5, sigma_range=25,
                     kernel_definition="diameter",
                     sigma_range_scale_to_dynamic_range=True):
    """对 float32 星图做双边滤波，返回 float32 结果。

    paper-confirmed: 滤波在对比度拉伸之前进行（论文流程顺序）。
    """
    d = kernel_diameter(kernel_size, kernel_definition)
    sigma_r = effective_sigma_range(
        image, sigma_range, sigma_range_scale_to_dynamic_range
    )
    src = np.ascontiguousarray(image, dtype=np.float32)
    out = cv2.bilateralFilter(src, d=d, sigmaColor=sigma_r, sigmaSpace=float(sigma_space))
    return out.astype(np.float32), {"d": d, "sigma_r_eff": sigma_r}
