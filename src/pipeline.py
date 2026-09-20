"""预处理流水线骨架（P1：读图 → 双边滤波 → SNR 评价）。

按 implementation_plan 的 01–09 编号输出中间图，当前实现步骤：
  01_original.png   原始帧（显示归一化）
  02_bilateral.png  双边滤波结果（显示归一化）
后续 P3/P4 将追加 03 拉伸 / 04-06 局部背景 / 07 二值化 / 08 形态学 / 09 最终图。

所有 python 调用使用 conda run -n HASD-StarNet。
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from .bilateral_filter import bilateral_filter
from .image_io import load_fits
from .metrics import calculate_snr
from .visualization import save_png


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_roi(roi_path):
    with open(roi_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_frame(cfg, roi, frame=None):
    """确定处理帧：显式指定 > ROI 参考帧 > 数据目录首帧。"""
    fits_dir = Path(cfg["data"]["fits_dir"])
    if frame is not None:
        return fits_dir / frame if not Path(frame).is_absolute() else Path(frame)
    ref = (roi or {}).get("reference_frame")
    if ref:
        return fits_dir / ref
    return sorted(fits_dir.glob("*.fits"))[0]


def process_frame(frame_path, cfg, roi):
    """对单帧执行 P1 流水线，返回 (结果字典, 中间图像字典)。"""
    image, stats = load_fits(str(frame_path))

    bcfg = cfg["bilateral"]
    filtered, binfo = bilateral_filter(
        image,
        kernel_size=bcfg["kernel_size"],
        sigma_space=bcfg["sigma_space"],
        sigma_range=bcfg["sigma_range"],
        kernel_definition=bcfg.get("kernel_definition", "diameter"),
        sigma_range_scale_to_dynamic_range=bcfg.get(
            "sigma_range_scale_to_dynamic_range", True
        ),
    )

    snr_before = calculate_snr(image, roi["target"], roi["background_annulus"])
    snr_after = calculate_snr(filtered, roi["target"], roi["background_annulus"])

    result = {
        "frame": Path(frame_path).name,
        "stats": stats,
        "bilateral": binfo,
        "snr_before": snr_before,
        "snr_after": snr_after,
    }
    images = {"01_original": image, "02_bilateral": filtered}
    return result, images


def run(config_path="config/paper.yaml", frame=None, out_dir=None):
    cfg = load_config(config_path)
    roi_path = cfg["data"].get("roi_file")
    roi = load_roi(roi_path) if roi_path else None
    if roi is None:
        raise ValueError("P1 验收需要 ROI（SNR 计算），请在配置中指定 data.roi_file")

    out_root = Path(out_dir or cfg["output"]["dir"])
    frame_path = resolve_frame(cfg, roi, frame)

    result, images = process_frame(frame_path, cfg, roi)
    for name, img in images.items():
        sub = "filtered" if "bilateral" in name else "debug"
        save_png(out_root / sub / f"{name}.png", img)

    print(f"frame: {result['frame']}")
    print(f"stats: {result['stats']}")
    bi = result["bilateral"]
    print(f"bilateral: d={bi['d']}, sigma_r_eff={bi['sigma_r_eff']:.4f}")
    sb, sa = result["snr_before"], result["snr_after"]
    print(f"SNR before: {sb['snr']:.3f} "
          f"(target_mean={sb['target_mean']:.1f}, bg_mean={sb['background_mean']:.1f}, "
          f"bg_std={sb['background_std']:.2f})")
    print(f"SNR after:  {sa['snr']:.3f} "
          f"(target_mean={sa['target_mean']:.1f}, bg_mean={sa['background_mean']:.1f}, "
          f"bg_std={sa['background_std']:.2f})")
    gain = (sa["snr"] - sb["snr"]) / sb["snr"] * 100.0
    print(f"SNR gain: {gain:+.2f}%")
    return result


def main():
    parser = argparse.ArgumentParser(description="星图预处理流水线（P1：双边滤波）")
    parser.add_argument("--config", default="config/paper.yaml")
    parser.add_argument("--frame", default=None, help="FITS 文件名或绝对路径；默认取 ROI 参考帧")
    parser.add_argument("--out", default=None, help="输出根目录；默认取配置 output.dir")
    args = parser.parse_args()
    run(args.config, frame=args.frame, out_dir=args.out)


if __name__ == "__main__":
    main()
