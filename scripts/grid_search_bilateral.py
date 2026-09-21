"""P2 双边滤波参数网格搜索。

paper-confirmed: 论文网格搜索范围 k∈{3,5,7}, sigma_space∈[0.5,2.5] step 0.5,
sigma_range∈[10,30] step 5（共 45 组，diameter 模式）。

流程：单帧（ROI 参考帧）→ 逐组合双边滤波 → 计算 SNR →
写 outputs/p2_grid_search/grid_search.csv，取 SNR 最大组合与论文 (5,1.5,25) 对照。
另以 radius 核定义（k=5 → 11x11）跑同一 sigma_space x sigma_range 网格作对照，
写 grid_search_radius.csv，辅助判断论文 k=5 的真实含义。

engineering-choice: 4096x4096 大图 x 多组合较慢，默认以 ROI 为中心裁剪
1024x1024 子图跑搜索（--crop-size 0 可切回全图）。注意裁剪后 sigma_r_eff
按子图动态范围缩放，与全图略有差异；结论用于参数排序，验收仍以全图为准。

用法：
    conda run -n HASD-StarNet python scripts/grid_search_bilateral.py
    conda run -n HASD-StarNet python scripts/grid_search_bilateral.py --crop-size 0
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import pandas as pd

from src.bilateral_filter import bilateral_filter
from src.image_io import load_fits
from src.metrics import calculate_snr
from src.pipeline import load_config, load_roi, resolve_frame

# paper-confirmed: 论文网格搜索范围
GRID_KERNEL_SIZES = [3, 5, 7]
GRID_SIGMA_SPACE = [0.5, 1.0, 1.5, 2.0, 2.5]
GRID_SIGMA_RANGE = [10, 15, 20, 25, 30]

# paper-confirmed: 论文报告的最优参数
PAPER_PARAMS = {"kernel_size": 5, "sigma_space": 1.5, "sigma_range": 25}

# engineering-choice: radius 模式对照只取 k=5（即 11x11），
# 与 diameter 网格共用 sigma_space x sigma_range 范围
RADIUS_KERNEL_SIZES = [5]


def crop_around_roi(image, roi, crop_size):
    """以 ROI 目标为中心裁剪 crop_size x crop_size 子图（越界自动收缩）。

    engineering-choice: P2 加速手段，见模块 docstring。
    """
    if crop_size <= 0:
        return image, (0, 0)
    h, w = image.shape
    cx, cy = int(roi["target"]["x"]), int(roi["target"]["y"])
    half = crop_size // 2
    x0 = int(np.clip(cx - half, 0, max(0, w - crop_size)))
    y0 = int(np.clip(cy - half, 0, max(0, h - crop_size)))
    x1, y1 = min(x0 + crop_size, w), min(y0 + crop_size, h)
    return image[y0:y1, x0:x1], (x0, y0)


def offset_roi(roi, x0, y0):
    """将 ROI 坐标平移到裁剪子图坐标系。"""
    return {
        "target": {**roi["target"], "x": roi["target"]["x"] - x0,
                   "y": roi["target"]["y"] - y0},
        "background_annulus": roi["background_annulus"],
    }


def search_grid(image, roi, kernel_sizes, kernel_definition):
    """对给定核尺寸集合跑 sigma_space x sigma_range 网格，返回结果行列表。"""
    scale = True  # 与 config/paper.yaml 的 sigma_range_scale_to_dynamic_range 一致
    snr_before = calculate_snr(image, roi["target"], roi["background_annulus"])
    rows = []
    for k in kernel_sizes:
        for ss in GRID_SIGMA_SPACE:
            for sr in GRID_SIGMA_RANGE:
                t0 = time.perf_counter()
                filtered, binfo = bilateral_filter(
                    image, kernel_size=k, sigma_space=ss, sigma_range=sr,
                    kernel_definition=kernel_definition,
                    sigma_range_scale_to_dynamic_range=scale,
                )
                runtime_ms = (time.perf_counter() - t0) * 1000.0
                snr_after = calculate_snr(
                    filtered, roi["target"], roi["background_annulus"]
                )
                rows.append({
                    "kernel_definition": kernel_definition,
                    "kernel_size": k,
                    "diameter_d": binfo["d"],
                    "sigma_space": ss,
                    "sigma_range": sr,
                    "sigma_r_eff": round(binfo["sigma_r_eff"], 4),
                    "snr_before": round(snr_before["snr"], 4),
                    "snr_after": round(snr_after["snr"], 4),
                    "snr_gain_percent": round(
                        (snr_after["snr"] - snr_before["snr"])
                        / snr_before["snr"] * 100.0, 2
                    ),
                    "target_mean": round(snr_after["target_mean"], 2),
                    "background_mean": round(snr_after["background_mean"], 2),
                    "background_std": round(snr_after["background_std"], 4),
                    "runtime_ms": round(runtime_ms, 1),
                })
    return rows, snr_before


def summarize(df, snr_before, paper_row_label):
    """打印 SNR 前 5 组合与论文参数对照，返回最优行。"""
    top = df.nlargest(5, "snr_after")
    print(f"\n[{paper_row_label}] SNR(before) = {snr_before['snr']:.3f}，Top-5 组合：")
    print(top[["kernel_size", "diameter_d", "sigma_space", "sigma_range",
               "snr_after", "snr_gain_percent"]].to_string(index=False))

    mask = ((df["kernel_size"] == PAPER_PARAMS["kernel_size"])
            & (df["sigma_space"] == PAPER_PARAMS["sigma_space"])
            & (df["sigma_range"] == PAPER_PARAMS["sigma_range"]))
    if mask.any():
        p = df[mask].iloc[0]
        print(f"论文参数 (k=5, σs=1.5, σr=25, d={int(p['diameter_d'])}): "
              f"SNR={p['snr_after']:.3f} ({p['snr_gain_percent']:+.2f}%)")
    best = df.loc[df["snr_after"].idxmax()]
    print(f"最优组合: k={int(best['kernel_size'])}, σs={best['sigma_space']}, "
          f"σr={int(best['sigma_range'])}, d={int(best['diameter_d'])}, "
          f"SNR={best['snr_after']:.3f} ({best['snr_gain_percent']:+.2f}%)")
    return best


def main():
    parser = argparse.ArgumentParser(description="P2 双边滤波参数网格搜索")
    parser.add_argument("--config", default="config/paper.yaml")
    parser.add_argument("--frame", default=None, help="FITS 文件名或绝对路径；默认取 ROI 参考帧")
    parser.add_argument("--crop-size", type=int, default=1024,
                        help="以 ROI 为中心的裁剪边长；0 表示全图")
    parser.add_argument("--out", default=None, help="输出根目录；默认 outputs，CSV 写到其下 p2_grid_search/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    roi = load_roi(cfg["data"]["roi_file"])
    frame_path = resolve_frame(cfg, roi, args.frame)
    out_dir = Path(args.out or "outputs") / "p2_grid_search"
    out_dir.mkdir(parents=True, exist_ok=True)

    image, stats = load_fits(str(frame_path))
    sub, (x0, y0) = crop_around_roi(image, roi, args.crop_size)
    sub_roi = offset_roi(roi, x0, y0)
    print(f"frame: {frame_path.name} ({stats})")
    print(f"search region: [{y0}:{y0 + sub.shape[0]}, {x0}:{x0 + sub.shape[1]}] "
          f"({sub.shape[1]}x{sub.shape[0]})")

    # diameter 模式：论文网格 k∈{3,5,7} x σs x σr（45 组）
    rows_d, snr_before = search_grid(sub, sub_roi, GRID_KERNEL_SIZES, "diameter")
    df_d = pd.DataFrame(rows_d).sort_values("snr_after", ascending=False)
    csv_d = out_dir / "grid_search.csv"
    df_d.to_csv(csv_d, index=False)
    print(f"已写出 {csv_d}（{len(df_d)} 组）")
    summarize(df_d, snr_before, "diameter")

    # radius 模式对照：k=5 → d=11（11x11），同一 σs x σr 网格（25 组）
    rows_r, _ = search_grid(sub, sub_roi, RADIUS_KERNEL_SIZES, "radius")
    df_r = pd.DataFrame(rows_r).sort_values("snr_after", ascending=False)
    csv_r = out_dir / "grid_search_radius.csv"
    df_r.to_csv(csv_r, index=False)
    print(f"已写出 {csv_r}（{len(df_r)} 组）")
    summarize(df_r, snr_before, "radius (k=5 → 11x11)")


if __name__ == "__main__":
    main()
