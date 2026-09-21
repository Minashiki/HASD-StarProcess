"""预处理流水线（P4：读图 → 双边滤波 → 对比度拉伸 → 局部背景建模
→ 自适应二值化 → 形态学 → SNR/检测指标评价）。

按 implementation_plan 的 01–09 编号输出中间图，按阶段落到子目录：
  p1_bilateral/          01_original.png    原始帧（显示归一化）
                         02_bilateral.png   双边滤波结果（显示归一化）
  p3_stretch_background/ 03_stretch.png     对比度拉伸到 [0,255]（minmax/statistical）
                         04_local_mean.png  20x20 块局部均值图 mu_loc
                         05_local_std.png   20x20 块局部标准差图 sigma_loc
                         06_S_map.png       显著性度量 S_map（论文公式 5）
  p4_binary_morphology/  07_binary.png      自适应二值化结果（reconstruction-assumption）
                         08_morphology.png  形态学处理结果（reconstruction-assumption）
                         09_final.png       最终前景掩膜叠加在拉伸图上的检测示意（BGR）

所有 python 调用使用 conda run -n HASD-StarNet。
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from .adaptive_threshold import threshold_strategy
from .bilateral_filter import bilateral_filter
from .contrast_stretch import contrast_stretch
from .image_io import load_fits
from .local_background import local_background_model
from .metrics import calculate_snr, detection_metrics
from .morphology import apply_morphology
from .visualization import overlay_mask, save_png


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
    """对单帧执行完整 P1–P4 流水线，返回 (结果字典, 中间图像字典)。"""
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
    stretch_method = ccfg.get("method", "minmax")
    stretch_info = {"method": stretch_method}
    stretch_kwargs = {}
    if stretch_method == "statistical":
        from .contrast_stretch import statistical_upper
        tier = ccfg.get("tier", "balanced")
        upper, median, sigma = statistical_upper(
            filtered, tier,
            percentile_tiers=ccfg.get("percentile_tiers"),
            upper_k=ccfg.get("upper_k", 8.0),
        )
        stretch_info.update(tier=tier, upper=upper,
                            background_median=median, background_sigma=sigma)
        stretch_kwargs = dict(tier=tier,
                              percentile_tiers=ccfg.get("percentile_tiers"),
                              upper_k=ccfg.get("upper_k", 8.0))
    stretched = contrast_stretch(
        filtered,
        out_min=ccfg.get("out_min", 0),
        out_max=ccfg.get("out_max", 255),
        method=stretch_method,
        **stretch_kwargs,
    )

    lbcfg = cfg.get("local_background", {})
    bg = local_background_model(stretched, block_size=lbcfg.get("block_size", 20))

    tcfg = cfg.get("adaptive_threshold", {})
    binary, t_map = threshold_strategy(
        stretched,
        bg["local_mean_map"],
        bg["S_map"],
        bg["local_std_map"],
        strategy=tcfg.get("strategy", "mu+C*S"),
        C=tcfg.get("C", 1.5),
    )

    mcfg = cfg.get("morphology", {})
    morphed = apply_morphology(
        binary,
        op=mcfg.get("op", "open"),
        kernel_shape=mcfg.get("kernel_shape", "ellipse"),
        kernel_size=mcfg.get("kernel_size", 3),
        iterations=mcfg.get("iterations", 1),
    )

    snr_before = calculate_snr(image, roi["target"], roi["background_annulus"])
    snr_filtered = calculate_snr(filtered, roi["target"], roi["background_annulus"])
    snr_after = calculate_snr(stretched, roi["target"], roi["background_annulus"])
    det_binary = detection_metrics(binary, stretched, roi["target"])
    det_final = detection_metrics(morphed, stretched, roi["target"])

    result = {
        "frame": Path(frame_path).name,
        "stats": stats,
        "bilateral": binfo,
        "contrast_stretch": stretch_info,
        "local_background": {
            "block_grid": bg["block_grid"],
            "global_std": bg["global_std"],
        },
        "threshold": {"strategy": tcfg.get("strategy", "mu+C*S"),
                      "C": tcfg.get("C", 1.5)},
        "morphology": dict(mcfg),
        "snr_before": snr_before,
        "snr_filtered": snr_filtered,
        "snr_after": snr_after,
        "det_binary": det_binary,
        "det_final": det_final,
    }
    images = {
        "01_original": image,
        "02_bilateral": filtered,
        "03_stretch": stretched,
        "04_local_mean": bg["local_mean_map"],
        "05_local_std": bg["local_std_map"],
        "06_S_map": bg["S_map"],
        "07_binary": binary,
        "08_morphology": morphed,
        "09_final": overlay_mask(stretched, morphed),
    }
    return result, images


def _stage_subdir(name):
    """按 01–09 编号确定阶段子目录（见模块 docstring）。"""
    stage = int(name.split("_")[0])
    if stage <= 2:      # 01 原图、02 双边滤波
        return "p1_bilateral"
    if stage <= 6:      # 03 拉伸、04-06 局部背景建模
        return "p3_stretch_background"
    return "p4_binary_morphology"  # 07 二值化、08 形态学、09 最终叠加


def _save_stage_images(out_root, images):
    """把 01–09 系列图按阶段子目录落盘到 out_root。"""
    for name, img in images.items():
        save_png(Path(out_root) / _stage_subdir(name) / f"{name}.png", img)


def run_batch(cfg, roi, out_root):
    """P5 批处理：对 fits_dir 全部帧跑完整流水线并汇总指标。

    engineering-choice: 每帧输出到 out_root/<帧名去扩展名>/ 下的阶段子目录；
    汇总指标写 out_root/batch_metrics.csv（pandas）。
    """
    import pandas as pd

    frames = sorted(Path(cfg["data"]["fits_dir"]).glob("*.fits"))
    if not frames:
        raise ValueError(f"批处理目录无 FITS 文件: {cfg['data']['fits_dir']}")
    out_root = Path(out_root)
    rows = []
    for i, fp in enumerate(frames, 1):
        result, images = process_frame(fp, cfg, roi)
        _save_stage_images(out_root / fp.stem, images)
        sb, sa = result["snr_before"], result["snr_after"]
        det = result["det_final"]
        rows.append({
            "frame": result["frame"],
            "snr_before": sb["snr"],
            "snr_filtered": result["snr_filtered"]["snr"],
            "snr_after": sa["snr"],
            "snr_gain_percent": (sa["snr"] - sb["snr"]) / sb["snr"] * 100.0,
            "target_energy_retention": det["target_energy_retention"],
            "background_noise_count": det["background_noise_count"],
            "foreground_pixels": det["foreground_pixels"],
        })
        print(f"[{i}/{len(frames)}] {result['frame']}: "
              f"SNR {sb['snr']:.2f} -> {sa['snr']:.2f} "
              f"({rows[-1]['snr_gain_percent']:+.1f}%), "
              f"energy_retention={det['target_energy_retention']:.3f}, "
              f"bg_noise={det['background_noise_count']}")
    csv_path = out_root / "batch_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.4f")
    print(f"saved: {csv_path}（{len(rows)} 帧）")
    return rows


def run(config_path="config/paper.yaml", frame=None, out_dir=None, batch=False,
        stretch_method=None, stretch_tier=None):
    cfg = load_config(config_path)
    if stretch_method is not None:
        cfg.setdefault("contrast_stretch", {})["method"] = stretch_method
    if stretch_tier is not None:
        cfg.setdefault("contrast_stretch", {})["tier"] = stretch_tier
    roi_path = cfg["data"].get("roi_file")
    roi = load_roi(roi_path) if roi_path else None
    if roi is None:
        raise ValueError("P1 验收需要 ROI（SNR 计算），请在配置中指定 data.roi_file")

    if batch:
        # engineering-choice: 批处理默认写到 output.dir/batch/，避免与单帧
        # 阶段目录混在一起；--out 可覆盖
        out_root = Path(out_dir) if out_dir else Path(cfg["output"]["dir"]) / "batch"
        return run_batch(cfg, roi, out_root)

    out_root = Path(out_dir or cfg["output"]["dir"])
    frame_path = resolve_frame(cfg, roi, frame)

    result, images = process_frame(frame_path, cfg, roi)
    _save_stage_images(out_root, images)

    print(f"frame: {result['frame']}")
    print(f"stats: {result['stats']}")
    bi = result["bilateral"]
    print(f"bilateral: d={bi['d']}, sigma_r_eff={bi['sigma_r_eff']:.4f}")
    cs = result["contrast_stretch"]
    if cs["method"] == "statistical":
        print(f"contrast_stretch: method=statistical, tier={cs['tier']}, "
              f"upper={cs['upper']:.1f} "
              f"(bg_median={cs['background_median']:.2f}, "
              f"bg_sigma={cs['background_sigma']:.2f})")
    else:
        print(f"contrast_stretch: method={cs['method']}")
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
    th = result["threshold"]
    print(f"threshold: strategy={th['strategy']}, C={th['C']}")
    mo = result["morphology"]
    print(f"morphology: op={mo.get('op', 'open')}, "
          f"{mo.get('kernel_shape', 'ellipse')} "
          f"{mo.get('kernel_size', 3)}x{mo.get('kernel_size', 3)}, "
          f"iter={mo.get('iterations', 1)}")
    for tag in ("det_binary", "det_final"):
        d = result[tag]
        print(f"{tag}: energy_retention={d['target_energy_retention']:.4f}, "
              f"pixel_retention={d['target_pixel_retention']:.4f}, "
              f"bg_noise_count={d['background_noise_count']}, "
              f"foreground_pixels={d['foreground_pixels']}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="星图预处理流水线（P4：双边滤波 + 对比度拉伸 + 局部背景建模"
                    " + 自适应二值化 + 形态学）"
    )
    parser.add_argument("--config", default="config/paper.yaml")
    parser.add_argument("--frame", default=None, help="FITS 文件名或绝对路径；默认取 ROI 参考帧")
    parser.add_argument("--out", default=None, help="输出根目录；默认取配置 output.dir")
    parser.add_argument("--batch", action="store_true",
                        help="批处理 fits_dir 全部帧（P5）；默认输出 output.dir/batch/")
    parser.add_argument("--stretch-method", choices=["minmax", "statistical"],
                        default=None,
                        help="覆盖配置中的对比度拉伸方法")
    parser.add_argument("--stretch-tier", choices=["recall", "balanced", "purity"],
                        default=None,
                        help="覆盖配置中的 statistical 拉伸百分位档位")
    args = parser.parse_args()
    run(args.config, frame=args.frame, out_dir=args.out, batch=args.batch,
        stretch_method=args.stretch_method, stretch_tier=args.stretch_tier)


if __name__ == "__main__":
    main()
