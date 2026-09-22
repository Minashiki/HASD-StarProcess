"""对比度拉伸模块（P3）。

paper-confirmed: 论文 1.1 节采用 min-max 线性拉伸到 [0,255]，
不使用 CLAHE / 百分位截断等非线性方法。

engineering-choice: 拉伸作用于双边滤波之后的 float32 数据（论文流程顺序：
滤波 → 拉伸 → 局部背景建模）。rst19 数据存在 0/65535 量级的坏点（uint16
重解释后的饱和/零值），全局 min-max 会被坏点主导（星点被压缩到低位灰度），
这与论文隐含假设一致、如实实现，其影响由后续局部背景建模与二值化缓解；
显示用 PNG 的百分位拉伸仅影响落盘图，不影响管线数据。

engineering-choice: 新增 statistical 方法（非论文方法，供对照实验选用）：
按统计范围确定灰度上限 upper = max(像素灰度百分位值, 背景中位数 + k×sigma)，
其中背景水平与噪声用 MAD（中位数绝对偏差）从输入图像（即 P1 双边滤波
结果）估计；大于 upper 的像素截断为 upper，再 min-max 归一化到
[out_min, out_max]。百分位档位（recall/balanced/purity）在配置中可调。

engineering-choice: statistical 方法支持 norm 开关（二次开发）：
norm=False 时只做高端截断、不做归一化，输出保持输入灰度量级
（截断后最大值 = upper），out_min/out_max 不生效。

engineering-choice: 新增 manual 方法（二次开发，灰度削峰）：
直接指定 manual_upper，高于它的像素截断为 manual_upper；
norm=True 时再 min-max 归一化到 [out_min, out_max]，norm=False 时
只截断、保留输入灰量级。与 statistical 的区别是上限由人工指定，
不做任何统计估计。
"""

import numpy as np

# engineering-choice: statistical 方法的默认百分位档位，配置可覆盖
DEFAULT_PERCENTILE_TIERS = {
    "recall": 99.5,
    "balanced": 99.9,
    "purity": 99.95,
}


def estimate_background_mad(image):
    """用 MAD 估计背景水平与噪声标准差，返回 (median, sigma)。

    median: 全图有限像素的中位数（背景水平）；
    mad:    各像素相对中位数的绝对偏差的中位数；
    sigma = 1.4826 * mad（MAD 到高斯标准差的换算系数）。
    """
    finite = np.asarray(image, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("图像无有限像素，无法估计背景")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median, 1.4826 * mad


def statistical_upper(image, tier="balanced", percentile_tiers=None, upper_k=8.0):
    """计算 statistical 方法的灰度上限，返回 (upper, median, sigma)。

    upper = max(像素灰度的 tier 百分位值, 背景中位数 + upper_k × sigma)。
    """
    tiers = dict(DEFAULT_PERCENTILE_TIERS)
    if percentile_tiers:
        tiers.update(percentile_tiers)
    if tier not in tiers:
        raise ValueError(f"未知百分位档位: {tier}（可选: {sorted(tiers)}）")
    img = np.asarray(image, dtype=np.float64)
    median, sigma = estimate_background_mad(img)
    pct = float(np.percentile(img[np.isfinite(img)], tiers[tier]))
    upper = max(pct, median + float(upper_k) * sigma)
    return upper, median, sigma


def contrast_stretch(image, out_min=0.0, out_max=255.0, method="minmax",
                     tier="balanced", percentile_tiers=None, upper_k=8.0,
                     norm=True, manual_upper=None):
    """对比度拉伸到 [out_min, out_max]，返回 float32。

    paper-confirmed: method="minmax" 为论文方法（全局 min-max 线性拉伸）。
    engineering-choice: method="statistical" 时先按 statistical_upper 截断
    高端灰度，再做 min-max 归一化，避免坏点主导动态范围；norm=False 时
    只做截断不做归一化，保留输入灰度量级（此时 out_min/out_max 不生效）。
    engineering-choice: method="manual" 为灰度削峰，截断上限由 manual_upper
    直接指定；norm 开关语义同 statistical。
    """
    img = np.asarray(image, dtype=np.float32)
    if method == "minmax":
        lo, hi = float(img.min()), float(img.max())
    elif method == "statistical":
        upper, _, _ = statistical_upper(img, tier, percentile_tiers, upper_k)
        img = np.minimum(img, np.float32(upper))
        if not norm:
            return img.astype(np.float32)
        lo, hi = float(img.min()), float(upper)
    elif method == "manual":
        if manual_upper is None:
            raise ValueError("method='manual' 需要指定 manual_upper")
        img = np.minimum(img, np.float32(manual_upper))
        if not norm:
            return img.astype(np.float32)
        lo, hi = float(img.min()), float(manual_upper)
    else:
        raise ValueError(
            f"未知拉伸方法: {method}（可选: minmax | statistical | manual）")
    if hi <= lo:
        return np.full(img.shape, float(out_min), dtype=np.float32)
    out = (img - lo) / (hi - lo) * (float(out_max) - float(out_min)) + float(out_min)
    return out.astype(np.float32)
