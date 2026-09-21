"""P0 辅助：选一颗孤立的中等亮度恒星作为 SNR 参考 ROI，写 data/roi/roi.yaml。"""
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.makedirs("data/roi", exist_ok=True)
os.makedirs("outputs/p0_debug", exist_ok=True)
warnings.filterwarnings("ignore")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.image_io import load_fits

a, _ = load_fits("rst19/20260330163205413_9901.fits")
H, W = a.shape
bg = np.median(a)

# 局部极大值：5x5 邻域最大值且等于自身、显著高于背景、远离饱和坏点(>30000)
dil = cv2.dilate(a, np.ones((5, 5), np.float32))
peaks = (a == dil) & (a > bg + 5 * a.std()) & (a < 30000)
peaks[:100, :] = peaks[-100:, :] = False
peaks[:, :100] = peaks[:, -100:] = False
ys, xs = np.where(peaks)
vals = a[ys, xs]

# 孤立性：100 px 内无其他峰
order = np.argsort(-vals)
pts = np.stack([xs[order], ys[order]], axis=1)
pvals = vals[order]
keep = []
suppressed = np.zeros(len(pts), bool)
for i in range(len(pts)):
    if suppressed[i]:
        continue
    keep.append(i)
    d2 = ((pts[:, 0] - pts[i, 0]) ** 2 + (pts[:, 1] - pts[i, 1]) ** 2)
    suppressed |= (d2 < 100 ** 2)
keep = np.array(keep)
iso_vals = pvals[keep]
iso_pts = pts[keep]

# 中等亮度：取孤立峰亮度中位数附近的一颗
med = np.median(iso_vals)
idx = np.argmin(np.abs(iso_vals - med))
sx, sy = iso_pts[idx]
print("isolated peaks:", len(iso_vals), "median peak:", med)
print("chosen star: x=%d y=%d peak=%.0f (bg=%.1f)" % (sx, sy, a[sy, sx], bg))

roi = {
    "description": (
        "SNR 参考 ROI。reconstruction-assumption: 序列中未发现明显的暗弱点状动目标"
        "（frame1 顶部的亮条带为卫星尾迹，且仅出现于单帧），按 implementation_plan P0 "
        "的备选方案选一颗孤立的中等亮度恒星。若拿到目标真值坐标可直接替换本文件。"
    ),
    "reference_frame": "20260330163205413_9901.fits",
    "target": {"x": int(sx), "y": int(sy), "radius": 5},
    "background_annulus": {"inner_radius": 10, "outer_radius": 20},
}
with open("data/roi/roi.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(roi, f, allow_unicode=True, sort_keys=False)
print("written data/roi/roi.yaml")

# 验证图：ROI 附近 200x200 zoom，叠加 target 圆与背景环
r = 100
crop = a[sy - r:sy + r, sx - r:sx + r]
l, h = np.percentile(crop, [1, 99.9])
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(np.clip((crop - l) / (h - l + 1e-9), 0, 1), cmap="gray")
for rr, c, lb in [(5, "r", "target"), (10, "y", "bg inner"), (20, "g", "bg outer")]:
    ax.add_patch(plt.Circle((r, r), rr, fill=False, color=c, label=lb))
ax.legend()
ax.set_title(f"ROI star ({sx},{sy})")
plt.tight_layout()
plt.savefig("outputs/p0_debug/p0_roi.png", dpi=90)
print("saved outputs/p0_debug/p0_roi.png")
