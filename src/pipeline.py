"""预处理流水线（P3：读图 → 双边滤波 → 对比度拉伸 → 局部背景建模 → SNR 评价）。

按 implementation_plan 的 01–09 编号输出中间图，当前实现步骤：
  01_original.png    原始帧（显示归一化）
  02_bilateral.png   双边滤波结果（显示归一化）
  03_stretch.png     min-max 对比度拉伸到 [0,255]
  04_local_mean.png  20x20 块局部均值图 mu_loc
  05_local_std.png   20x20 块局部标准差图 sigma_loc
  06_S_map.png       显著性度量 S_map（论文公式 5）
后续 P4 将追加 07 二值化 / 08 形态学 / 09 最终图。

所有 python 调用使用 conda run -n HASD-StarNet。
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from .bilateral_filter import bilateral_filter
from .contrast_stretch import contrast_stretch
from .image_io import load_fits
from .local_background import local_background_model
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
    """对单帧执行 P3 流水线，返回 (结果字典, 中间图像字典)。"""
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

    ccfg = cfg.get("contrast_stretch", {})
    stretched = contrast_stretch(
        filtered,
        out_min=ccfg.get("out_min", 0),
        out_max=ccfg.get("out_max", 255),
        method=ccfg.get("method", "minmax"),
    )

    lbcfg = cfg.get("local_background", {})
    bg = local_background_model(stretched, block_size=lbcfg.get("block_size", 20))

    snr_before = calculate_snr(image, roi["target"], roi["background_annulus"])
    snr_filtered = calculate_snr(filtered, roi["target"], roi["background_annulus"])
    snr_after = calculate_snr(stretched, roi["target"], roi["background_annulus"])

    result = {
        "frame": Path(frame_path).name,
        "stats": stats,
        "bilateral": binfo,
        "local_background": {
            "block_grid": bg["block_grid"],
            "global_std": bg["global_std"],
        },
        "snr_before": snr_before,
        "snr_filtered": snr_filtered,
        "snr_after": snr_after,
    }
    images = {
        "01_original": image,
        "02_bilateral": filtered,
        "03_stretch": stretched,
        "04_local_mean": bg["local_mean_map"],
        "05_local_std": bg["local_std_map"],
        "06_S_map": bg["S_map"],
    }
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
    lb = result["local_background"]
    print(f"local_background: block_grid={lb['block_grid']}, "
          f"global_std={lb['global_std']:.3f}")
    sb, sa = result["snr_before"], result["snr_after"]
    sf = result["snr_filtered"]
    print(f"SNR before:   {sb['snr']:.3f} "
          f"(target_mean={sb['target_mean']:.1f}, bg_mean={sb['background_mean']:.1f}, "
          f"bg_std={sb['background_std']:.2f})")
    print(f"SNR filtered: {sf['snr']:.3f} "
          f"(target_mean={sf['target_mean']:.1f}, bg_mean={sf['background_mean']:.1f}, "
          f"bg_std={sf['background_std']:.2f})")
    print(f"SNR stretch:  {sa['snr']:.3f} "
          f"(target_mean={sa['target_mean']:.1f}, bg_mean={sa['background_mean']:.1f}, "
          f"bg_std={sa['background_std']:.2f})")
    gain = (sa["snr"] - sb["snr"]) / sb["snr"] * 100.0
    print(f"SNR gain (stretch vs original): {gain:+.2f}%")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="星图预处理流水线（P3：双边滤波 + 对比度拉伸 + 局部背景建模）"
    )
    parser.add_argument("--config", default="config/paper.yaml")
    parser.add_argument("--frame", default=None, help="FITS 文件名或绝对路径；默认取 ROI 参考帧")
    parser.add_argument("--out", default=None, help="输出根目录；默认取配置 output.dir")
    args = parser.parse_args()
    run(args.config, frame=args.frame, out_dir=args.out)


if __name__ == "__main__":
    main()
