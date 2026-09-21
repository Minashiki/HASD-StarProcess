"""P5-D 滤波器对照实验：mean / median / gaussian / bilateral。

paper-confirmed: 论文预处理采用双边滤波（公式 (1)-(4)，k=5, sigma_space=1.5,
sigma_range=25），未涉及其他滤波器。
engineering-choice: 对照滤波器（mean/median/gaussian）取与双边滤波相同的
空间支持（核边长 = bilateral.kernel_size，高斯 sigma = bilateral.sigma_space），
保证对照公平。
engineering-choice: SNR = |m - m_b| / sigma_b 对线性变换不变，故对照直接在
滤波结果上计算 SNR，与"滤波 -> min-max 拉伸"后的 SNR 相等（浮点舍入内），
不再重复拉伸。

用法：
    conda run -n HASD-StarNet python scripts/compare_filters.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bilateral_filter import bilateral_filter
from src.metrics import calculate_snr
from src.pipeline import load_config, load_roi, resolve_frame
from src.image_io import load_fits
from src.visualization import normalize_to_uint8

FILTERS = ("mean", "median", "gaussian", "bilateral")


def apply_filter(image, name, bcfg):
    """按名称对 float32 图像滤波，返回 (滤波结果 float32, runtime_ms)。

    engineering-choice: mean/median/gaussian 的核参数跟随双边滤波配置
    （同空间支持对照）；bilateral 走 src.bilateral_filter（含 sigma_range
    动态范围缩放）。
    """
    k = int(bcfg["kernel_size"])
    src = np.ascontiguousarray(image, dtype=np.float32)
    t0 = time.perf_counter()
    if name == "mean":
        out = cv2.blur(src, (k, k))
    elif name == "median":
        # cv2.medianBlur 对 CV_32F 仅支持 ksize 3/5；配置 k>5 时会报错
        out = cv2.medianBlur(src, k)
    elif name == "gaussian":
        out = cv2.GaussianBlur(src, (k, k), sigmaX=float(bcfg["sigma_space"]))
    elif name == "bilateral":
        out, _ = bilateral_filter(
            src,
            kernel_size=k,
            sigma_space=bcfg["sigma_space"],
            sigma_range=bcfg["sigma_range"],
            kernel_definition=bcfg.get("kernel_definition", "diameter"),
            sigma_range_scale_to_dynamic_range=bcfg.get(
                "sigma_range_scale_to_dynamic_range", True
            ),
        )
    else:
        raise ValueError(f"未知滤波器 {name!r}，候选: {', '.join(FILTERS)}")
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return out.astype(np.float32), runtime_ms


def run_filter_comparison(image, cfg, roi):
    """对单帧跑 D 组四种滤波对照，返回 (指标行列表, 滤波结果图像字典)。"""
    bcfg = cfg["bilateral"]
    rows, images = [], {}
    for name in FILTERS:
        filtered, runtime_ms = apply_filter(image, name, bcfg)
        s = calculate_snr(filtered, roi["target"], roi["background_annulus"])
        rows.append({
            "group": "D",
            "method": name,
            "snr_after": s["snr"],
            "background_mean": s["background_mean"],
            "background_std": s["background_std"],
            "target_mean": s["target_mean"],
            "runtime_ms": runtime_ms,
        })
        images[name] = filtered
    return rows, images


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config("config/paper.yaml")
    roi = load_roi(cfg["data"]["roi_file"])
    frame_path = resolve_frame(cfg, roi)
    image, _ = load_fits(str(frame_path))
    rows, images = run_filter_comparison(image, cfg, roi)

    out_dir = Path("outputs/p5_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "compare_filters.csv", index=False)

    fig, axes = plt.subplots(1, len(FILTERS), figsize=(4 * len(FILTERS), 4.5))
    for ax, row in zip(axes, rows):
        ax.imshow(normalize_to_uint8(images[row["method"]]), cmap="gray")
        ax.set_title(f"{row['method']}\nSNR={row['snr_after']:.2f}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_filters.png", dpi=150, bbox_inches="tight",
                pad_inches=0.1)
    plt.close(fig)

    for row in rows:
        print(f"{row['method']:>9}: SNR={row['snr_after']:8.3f}  "
              f"bg_std={row['background_std']:.3f}  "
              f"target_mean={row['target_mean']:.1f}  "
              f"runtime={row['runtime_ms']:.0f} ms")
    print(f"saved: {out_dir / 'compare_filters.csv'}, {out_dir / 'compare_filters.png'}")


if __name__ == "__main__":
    main()
