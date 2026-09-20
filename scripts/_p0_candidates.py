"""P0 辅助：帧差找运动目标候选，输出候选点 zoom 图供人工选 ROI。"""
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

f1 = "rst19/20260330163205413_9901.fits"
f2 = "rst19/20260330163226427_9901.fits"
a, _ = load_fits(f1)
b, _ = load_fits(f2)
d = b.astype(np.float64) - a.astype(np.float64)
dm = np.abs(d)

# 候选点：|diff| 局部大值，互相间隔 >= 80 px
work = dm.copy()
cands = []
for _ in range(30):
    idx = np.argmax(work)
    y, x = divmod(idx, work.shape[1])
    v = work[y, x]
    if v < 500:
        break
    cands.append((int(x), int(y), float(v), float(d[y, x])))
    work[max(0, y - 80):y + 80, max(0, x - 80):x + 80] = 0

print("top candidates (x, y, |diff|, signed_diff):")
for c in cands:
    print(c)

# 每个候选渲染 200x200 zoom：frame1 / frame15 / diff
n = min(len(cands), 12)
fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
for i, (x, y, v, sd) in enumerate(cands[:n]):
    r = 100
    y0, y1 = max(0, y - r), min(a.shape[0], y + r)
    x0, x1 = max(0, x - r), min(a.shape[1], x + r)
    for j, (img, ttl, cmap) in enumerate(
        [(a, "f1", "gray"), (b, "f15", "gray"), (d, "diff", "RdBu")]
    ):
        ax = axes[i, j]
        crop = img[y0:y1, x0:x1]
        if cmap == "gray":
            l, h = np.percentile(crop, [1, 99.9])
            ax.imshow(np.clip((crop - l) / (h - l + 1e-9), 0, 1), cmap=cmap)
        else:
            m = np.percentile(np.abs(crop), 99.9)
            ax.imshow(crop, cmap=cmap, vmin=-m, vmax=m)
        ax.set_title(f"#{i} ({x},{y}) {ttl} |d|={v:.0f}")
        ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/debug/p0_candidates.png", dpi=70)
print("saved outputs/debug/p0_candidates.png")
