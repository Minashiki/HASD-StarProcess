"""P5 SNR 评价与四组实验（implementation_plan P5）。

四组实验（参考帧，ROI 见 data/roi/roi.yaml）：
  A 原图：不处理，SNR 基线。
  B 双边滤波：论文参数 (k=5, sigma_space=1.5, sigma_range=25)，diameter 模式。
  C 完整 pipeline：双边滤波 -> min-max 拉伸 -> 局部背景建模 -> 自适应二值化
    -> 形态学。SNR 在拉伸图上计算（engineering-choice：SNR 对线性拉伸不变，
    与滤波后 SNR 相等）；另报告形态学结果的目标能量保留率 / 背景噪点数
    （P4 detection_metrics，engineering-choice）。
  D 滤波器对照：mean / median / gaussian 与双边滤波同空间支持对照
    （见 scripts/compare_filters.py）。

输出（outputs/p5_evaluation/，engineering-choice 目录名）：
  metrics.csv     列：image, method, snr_before, snr_after, snr_gain_percent,
                  background_mean, background_std, target_mean, runtime_ms
                  （implementation_plan P5 规定），附加 group, target_peak,
                  energy_retention, background_noise_count（判据辅助列）。
  comparison.png  原图 + D 组四滤波结果，五图并排（显示归一化，标注 SNR）。

判据（implementation_plan P5）：SNR_after > SNR_before 且目标峰值 / 局部能量
未明显破坏；不硬性要求论文的 93.37% 提升。

用法：
    conda run -n HASD-StarNet python scripts/evaluate_preprocess.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_filters import FILTERS, run_filter_comparison

from src.metrics import calculate_snr
from src.pipeline import load_config, load_roi, process_frame, resolve_frame
from src.image_io import load_fits
from src.visualization import normalize_to_uint8

OUT_DIR = Path("outputs/p5_evaluation")

# engineering-choice: 目标圆盘 target_mean 保留率低于该阈值视为"目标被明显
# 破坏"（固定圆盘内平均灰度正比于局部能量；单像素峰值仅作参考）
TARGET_KEEP_MIN = 0.8


def target_peak(image, target):
    """目标圆盘内峰值灰度（判据辅助量，engineering-choice）。"""
    cx, cy, r = int(target["x"]), int(target["y"]), float(target["radius"])
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    return float(image[mask].max())


def _row(group, method, image_name, snr_before, s, runtime_ms, peak,
         energy_retention=None, noise_count=None):
    return {
        "group": group,
        "image": image_name,
        "method": method,
        "snr_before": snr_before,
        "snr_after": s["snr"],
        "snr_gain_percent": (s["snr"] - snr_before) / snr_before * 100.0,
        "background_mean": s["background_mean"],
        "background_std": s["background_std"],
        "target_mean": s["target_mean"],
        "target_peak": peak,
        "runtime_ms": runtime_ms,
        "energy_retention": energy_retention,
        "background_noise_count": noise_count,
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config("config/paper.yaml")
    roi = load_roi(cfg["data"]["roi_file"])
    frame_path = resolve_frame(cfg, roi)
    image_name = Path(frame_path).name
    image, _ = load_fits(str(frame_path))

    peak_orig = target_peak(image, roi["target"])
    s_orig = calculate_snr(image, roi["target"], roi["background_annulus"])
    snr_before = s_orig["snr"]
    rows = [_row("A", "original", image_name, snr_before, s_orig, 0.0, peak_orig)]

    # B + D：D 组对照包含 bilateral，与 B 为同一结果，B 行复用 D 的数值
    d_rows, d_images = run_filter_comparison(image, cfg, roi)
    b = next(r for r in d_rows if r["method"] == "bilateral")
    b_snr = {"snr": b["snr_after"], "background_mean": b["background_mean"],
             "background_std": b["background_std"], "target_mean": b["target_mean"]}
    rows.append(_row("B", "bilateral", image_name, snr_before, b_snr,
                     b["runtime_ms"], target_peak(d_images["bilateral"], roi["target"])))
    for r in d_rows:
        if r["method"] == "bilateral":
            continue  # 已在 B 行给出，D 组内不重复
        r_snr = {"snr": r["snr_after"], "background_mean": r["background_mean"],
                 "background_std": r["background_std"], "target_mean": r["target_mean"]}
        rows.append(_row("D", r["method"], image_name, snr_before, r_snr,
                         r["runtime_ms"], target_peak(d_images[r["method"]], roi["target"])))

    # C 完整 pipeline（target_peak 取拉伸图 [0,255] 尺度的峰值，与原图尺度
    # 不同，不参与 peak_ratio 判据；C 的目标保留判据用能量保留率）
    t0 = time.perf_counter()
    result, pipe_images = process_frame(frame_path, cfg, roi)
    pipeline_ms = (time.perf_counter() - t0) * 1000.0
    sa = result["snr_after"]
    det = result["det_final"]
    rows.append(_row("C", "pipeline_full", image_name, snr_before, sa,
                     pipeline_ms, target_peak(pipe_images["03_stretch"], roi["target"]),
                     energy_retention=det["target_energy_retention"],
                     noise_count=det["background_noise_count"]))

    df = pd.DataFrame(rows, columns=[
        "image", "method", "snr_before", "snr_after", "snr_gain_percent",
        "background_mean", "background_std", "target_mean", "runtime_ms",
        "group", "target_peak", "energy_retention", "background_noise_count",
    ])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "metrics.csv", index=False, float_format="%.4f")

    # comparison.png：原图 + D 组四滤波结果，五图并排
    panels = [("A original", image, snr_before)]
    panels += [(f"D {name}", d_images[name],
                next(r["snr_after"] for r in d_rows if r["method"] == name))
               for name in FILTERS]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.5))
    for ax, (title, img, snr) in zip(axes, panels):
        ax.imshow(normalize_to_uint8(img), cmap="gray")
        ax.set_title(f"{title}\nSNR={snr:.2f}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison.png", dpi=150, bbox_inches="tight",
                pad_inches=0.1)
    plt.close(fig)

    # 判据检查（implementation_plan P5：SNR_after > SNR_before 且目标峰值/
    # 局部能量未明显破坏）
    # engineering-choice: "局部能量未明显破坏"用 target_mean 保留率判定
    # （固定目标圆盘内的平均灰度正比于局部能量；单像素峰值对点源平滑天然
    # 敏感，仅作参考列）。overall 只就论文方法链路（B 双边滤波、C 完整
    # pipeline）判定；D 组为对照实验，各自的 PASS/FAIL 是对照结论的一部分
    # （例如 median 摧毁星点正说明其不适合）。
    mean_orig = s_orig["target_mean"]
    print(df.to_string(index=False))
    print("\n== P5 判据 ==")
    ok = True
    for r in rows:
        if r["method"] == "original":
            continue
        snr_ok = r["snr_after"] > snr_before
        msg = f"{r['method']:>13}: SNR {snr_before:.2f} -> {r['snr_after']:.2f} " \
              f"({'PASS' if snr_ok else 'FAIL'}"
        if r["method"] == "pipeline_full":
            er = r["energy_retention"]
            er_ok = er is not None and er > 0
            ok &= snr_ok and er_ok
            msg += f", energy_retention={er:.3f} {'PASS' if er_ok else 'FAIL'}"
        else:
            mean_ratio = r["target_mean"] / mean_orig if mean_orig > 0 else 0.0
            mean_ok = mean_ratio >= TARGET_KEEP_MIN
            peak_ratio = r["target_peak"] / peak_orig if peak_orig > 0 else 0.0
            if r["group"] == "B":
                ok &= snr_ok and mean_ok
            msg += (f", target_mean_ratio={mean_ratio:.3f} "
                    f"{'PASS' if mean_ok else 'FAIL'}"
                    f", peak_ratio={peak_ratio:.3f}(ref)")
        msg += ")"
        print(msg)
    print(f"overall (B/C 论文方法链路): {'PASS' if ok else 'FAIL'}"
          f"（不硬性要求论文 93.37% 提升）")
    print(f"saved: {OUT_DIR / 'metrics.csv'}, {OUT_DIR / 'comparison.png'}")


if __name__ == "__main__":
    main()
