"""P3 模块快速 sanity check（临时脚本，验证后可删除）。"""

import numpy as np

from src.contrast_stretch import contrast_stretch
from src.local_background import block_statistics, compute_s_map, local_background_model

# 拉伸正确性
a = np.array([[10.0, 20.0], [30.0, 50.0]], dtype=np.float32)
s = contrast_stretch(a)
assert s.min() == 0.0 and s.max() == 255.0, (s.min(), s.max())
assert np.isclose(s[0, 1], (20 - 10) / 40 * 255), s[0, 1]

# 块统计正确性（40x40, 4 个 20x20 块，各块取常数 -> std=0）
b = np.zeros((40, 40), dtype=np.float32)
b[:20, :20] = 5.0
b[:20, 20:] = 10.0
b[20:, :20] = 15.0
b[20:, 20:] = 20.0
st = block_statistics(b, 20)
assert st["block_grid"] == (2, 2)
assert st["block_mean"].tolist() == [[5.0, 10.0], [15.0, 20.0]]
assert (st["block_std"] == 0).all()
assert st["local_mean_map"][0, 0] == 5.0 and st["local_mean_map"][39, 39] == 20.0

# 残块（30 列 -> 20 + 10）
c = np.ones((20, 30), dtype=np.float32)
st2 = block_statistics(c, 20)
assert st2["block_grid"] == (1, 2)
assert st2["block_mean"][0, 1] == 1.0

# S_map 公式 (5): S = [1 - exp(-sl^2/sg^2)] * sg
sl = np.array([[1.0, 2.0]], dtype=np.float32)
sm = compute_s_map(sl, 2.0)
exp0 = (1 - np.exp(-1.0 / 4.0)) * 2.0
assert np.isclose(sm[0, 0], exp0), (sm[0, 0], exp0)
assert sm[0, 1] > sm[0, 0]

# 端到端
rng = np.random.default_rng(0)
img = rng.normal(100, 5, (100, 100)).astype(np.float32)
m = local_background_model(img, 20)
assert m["S_map"].shape == (100, 100)
assert m["global_std"] > 0
print("all sanity checks passed")
