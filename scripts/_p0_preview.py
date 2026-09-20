"""P0 辅助：渲染首/末帧与帧差图，用于人工选取 ROI。"""
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
a, sa = load_fits(f1)
b, sb = load_fits(f2)
print("frame1:", sa)
print("frame15:", sb)


def stretch(x, lo=0.5, hi=99.5):
    l, h = np.percentile(x, [lo, hi])
    return np.clip((x - l) / (h - l + 1e-9), 0, 1)


d = b.astype(np.float64) - a.astype(np.float64)
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
axes[0].imshow(stretch(a)[::4, ::4], cmap="gray")
axes[0].set_title("frame 1")
axes[1].imshow(stretch(b)[::4, ::4], cmap="gray")
axes[1].set_title("frame 15")
dm = np.abs(d)
axes[2].imshow(np.clip(dm[::4, ::4] / np.percentile(dm, 99.9), 0, 1), cmap="hot")
axes[2].set_title("|diff| f15-f1")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/debug/p0_overview.png", dpi=80)
print("saved outputs/debug/p0_overview.png")
print("diff p99.9:", np.percentile(dm, 99.9), "diff max:", dm.max())
