# StarImgPro02 — HASD-StarNet 星图预处理复现

复现论文《复杂星空背景下的暗弱空间目标层次化检测框架》（袁博）第 1.1 节的星图预处理流程：
自适应双边滤波 → 对比度拉伸 → 局部背景建模 → 自适应二值化 → 形态学处理，并以 SNR 定量评价。

## 复现标注原则

代码注释与文档中显式区分三类内容：

- **paper-confirmed**：论文明确给出（公式、参数值），直接实现。
- **engineering-choice**：论文未规定、实现时需要做的工程决策（如数据类型、缩放方式）。
- **reconstruction-assumption**：论文缺失或含糊、复现时必须自行假设的内容（如二值化判决式、形态学参数）。

## 三个不确定项（论文缺失，重建假设）

1. **二值化判决式**：1.1 节未给出显式 T(x,y) 判决公式。重建默认 `T = μloc + C·S`（S 为公式 (5) 的显著性度量），C 可配；`T = μloc + C·σloc` 作为对照（参考论文 1.2.1 节 C=1.5）。全部标记 `reconstruction-assumption`。
2. **形态学参数**：论文仅在图 2 中出现形态学，未给参数。默认 `open, ellipse, 3×3, iter 1`，按星点能量保留率与背景噪点数选取。
3. **σr 灰度尺度**：论文 σr=25 隐含 [0,255] 灰度尺度，但滤波在拉伸之前作用于原始 float 数据，需按动态范围缩放（可配置，`engineering-choice`）。

## 环境

```bash
conda run -n HASD-StarNet pip install -r requirements.txt
```

所有 python/pip 调用统一使用 `conda run -n HASD-StarNet ...`（Python 3.12.14）。

## 数据

`rst19/`：15 帧 FITS 序列，BITPIX=16（有符号 int16），4096×4096，曝光 1500 ms，IMAGETYP=OBJECT。
读取后内部统一转 float32。

## 目录结构

见 `implementation_plan.md` 第二节。`src/` 为预处理模块，`scripts/` 为实验脚本，
`tests/` 为单元测试，`outputs/`（gitignore）存放中间图与指标。

## 运行

```bash
conda run -n HASD-StarNet python scripts/run_preprocess.py --config config/paper.yaml
conda run -n HASD-StarNet python -m pytest tests/
```

## 进度

- **P0 环境与工程**（完成）：目录结构、配置、FITS 读取（`src/image_io.py`）、ROI 选取。
- **P1 双边滤波**（完成）：`src/bilateral_filter.py`（cv2.bilateralFilter，diameter/radius
  两种核定义，σr 按帧动态范围缩放）、`src/metrics.py`（SNR=|m−m_b|/σ_b）、
  `src/visualization.py`、`src/pipeline.py` 骨架、`scripts/run_preprocess.py`。
  参考帧验收：SNR 17.61 → 43.37（+146.25%），目标均值 202.4 → 199.0（保留），
  背景 σ 10.31 → 4.11。输出 `outputs/debug/01_original.png`、`outputs/filtered/02_bilateral.png`。

显示归一化说明（engineering-choice）：PNG 落盘用 0.5%/99.5% 百分位拉伸
（±32768 量级坏点会使全局 min-max 把星点压到不可见）；管线内数据不受影响。

## ROI

`data/roi/roi.yaml`：SNR 参考 ROI，target=(1448, 1636, r=5)，背景环 inner=10 / outer=20。

**ROI 选取说明（reconstruction-assumption）**：P0 阶段对首/末帧做帧差检查后，未发现明显的暗弱
点状动目标——frame 1 顶部 (1448,370) 附近的亮条带为卫星尾迹，仅出现于单帧，不适合做点状
SNR ROI；帧差中其余 ±6×10⁴ 量级的散点为闪烁坏点。按 implementation_plan P0 的备选方案，
选取一颗孤立的中等亮度恒星（峰值 3993，背景 21）作为 SNR 参考。若拿到目标真值坐标可直接
替换 `data/roi/roi.yaml`。
