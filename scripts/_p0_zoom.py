"""P0 辅助：zoom 条带区域（疑似运动目标）与一颗中等亮度恒星候选。"""
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.makedirs("outputs/debug", exist_ok=True)
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.image_io import load_fits

a, _ = load_fits("rst19/20260330163205413_9901.fits")
b, _ = load_fits("rst19/20260330163226427_9901.fits")

# 条带区域（overview 中约在 x 1300-1600, y 200-500）
regions = {"streak": (1400, 350, 500), "star_field": (2048, 2048, 400)}
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for row, (name, (cx, cy, r)) in enumerate(regions.items()):
    y0, y1 = max(0, cy - r), cy + r
    x0, x1 = max(0, cx - r), cx + r
    d = b.astype(np.float64) - a.astype(np.float64)
    for col, (img, ttl) in enumerate([(a, "f1"), (b, "f15"), (d, "diff")]):
        ax = axes[row, col]
        crop = img[y0:y1, x0:x1]
        if ttl == "diff":
            m = np.percentile(np.abs(crop), 99.5)
            ax.imshow(crop, cmap="RdBu", vmin=-m, vmax=m)
        else:
            l, h = np.percentile(crop, [1, 99.9])
            ax.imshow(np.clip((crop - l) / (h - l + 1e-9), 0, 1), cmap="gray")
        ax.set_title(f"{name} ({cx},{cy})±{r} {ttl}")
        ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/debug/p0_zoom.png", dpi=90)
print("saved outputs/debug/p0_zoom.png")

# 条带区域精确坐标：frame1 条带内亮点质心
crop = a[150:600, 1200:1700]
thr = np.percentile(crop, 99.9)
ys, xs = np.where(crop > thr)
print("streak bright pixels centroid (f1): x=%.0f y=%.0f, n=%d, thr=%.0f"
      % (xs.mean() + 1200, ys.mean() + 150, len(xs), thr))
crop2 = b[150:600, 1200:1700]
ys2, xs2 = np.where(crop2 > thr)
print("streak bright pixels centroid (f15): x=%.0f y=%.0f, n=%d"
      % (xs2.mean() + 1200, ys2.mean() + 150, len(xs2)))
