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

`outputs/` 下按阶段命名子目录：`p0_debug/`（P0 选 ROI 辅助图）、
`p1_bilateral/`（01–02）、`p3_stretch_background/`（03–06）、
`p4_binary_morphology/`（07–09）、`p2_grid_search/`（P2 网格搜索 CSV）、
`p5_evaluation/`（P5 metrics.csv 与 comparison.png）。
`--out` 指定自定义输出根目录（如 `outputs/test1`）时，其下也按同样的
p 阶段子目录结构生成。`--batch` 批处理默认写到 `outputs/batch/<帧名>/`
（每帧一套 01–09 阶段子目录）并汇总 `outputs/batch/batch_metrics.csv`。

## 运行

```bash
conda run -n HASD-StarNet python scripts/run_preprocess.py --config config/paper.yaml
conda run -n HASD-StarNet python scripts/run_preprocess.py --batch     # P5 全 15 帧批处理
conda run -n HASD-StarNet python scripts/grid_search_bilateral.py      # P2 网格搜索
conda run -n HASD-StarNet python scripts/evaluate_preprocess.py        # P5 四组实验
conda run -n HASD-StarNet python scripts/compare_filters.py            # P5-D 滤波器对照（可独立运行）
conda run -n HASD-StarNet python -m pytest tests/
```

注意（engineering-choice）：Windows 下脚本输出含中文/希腊字母时，conda run 按 UTF-8
解码子进程输出，需前置 `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`，否则 conda 端
GBK 打印报 UnicodeEncodeError。

## 进度

- **P0 环境与工程**（完成）：目录结构、配置、FITS 读取（`src/image_io.py`）、ROI 选取。
- **P1 双边滤波**（完成）：`src/bilateral_filter.py`（cv2.bilateralFilter，diameter/radius
  两种核定义，σr 按帧动态范围缩放）、`src/metrics.py`（SNR=|m−m_b|/σ_b）、
  `src/visualization.py`、`src/pipeline.py` 骨架、`scripts/run_preprocess.py`。
  参考帧验收：SNR 17.61 → 43.37（+146.25%），目标均值 202.4 → 199.0（保留），
  背景 σ 10.31 → 4.11。输出 `outputs/p1_bilateral/01_original.png`、
  `outputs/p1_bilateral/02_bilateral.png`。
- **P2 网格搜索**（完成）：`scripts/grid_search_bilateral.py`，论文网格
  k∈{3,5,7} × σs∈{0.5..2.5} × σr∈{10..30}（3×5×5=75 组，diameter 模式）
  + radius 模式（k=5 → 11×11）25 组对照，单帧 ROI 中心 1024×1024 裁剪加速
  （engineering-choice：σr_eff 按子图动态范围缩放，结论用于参数排序）。
  输出 `outputs/p2_grid_search/grid_search.csv`、`grid_search_radius.csv`。
  结果（SNR_before=17.61）：
  - diameter：论文参数 (5, 1.5, 25) SNR=43.32（+145.93%）；网格最优 (7, 2.5, 30)
    SNR=60.09（+241.17%），SNR 随 k/σs/σr 增大单调上升，最优落在网格边界。
  - radius（k=5 → 11×11）：论文参数 SNR=59.07（+235.33%）；最优 (2.5, 30)
    SNR=79.30（+350.21%），整体高于同参数 diameter。
  - 结论：SNR 指标奖励背景平滑，网格最优顶到边界，不具区分度；论文 (5,1.5,25)
    应理解为"足够平滑且保留目标"的折中。k=5 按 radius（11×11）解释时同参数
    SNR 更高，但两种解释均无法仅凭 SNR 排除，留待 P5 四组实验结合目标能量
    保留情况综合判断。
- **P3 对比度拉伸 + 局部背景模型**（完成）：`src/contrast_stretch.py`
  （paper-confirmed min-max → [0,255]）、`src/local_background.py`
  （20×20 块统计 μloc/σloc 图、σglo，公式 (5) S_map，边缘残块按实际像素
  统计，块图最近邻放大回原尺寸）。`src/pipeline.py` 串联 P1–P3，
  输出 `03_stretch.png`、`04_local_mean.png`、`05_local_std.png`、`06_S_map.png`。
  参考帧验收：块网格 205×205，σglo=0.650；拉伸后 SNR=43.372 与滤波后
  43.374 一致（min-max 为线性变换，SNR 不变，浮点舍入内）。
  注意（paper-faithful 后果）：rst19 存在 ±32768 量级坏点，全局 min-max
  把有效信号压缩到很窄灰度区间（背景 σ 仅 0.02/255），S_map 仍能正常
  分离星点块（见 `06_S_map.png`），后续 P4 阈值在此窄区间数据上按
  相对统计量工作。
- **P4 自适应二值化 + 形态学**（完成）：`src/adaptive_threshold.py`
  （三种候选策略 `mu+S` / `mu+C*S` / `mu+C*sigma_loc`，判决 image ≥ T，
  全部 reconstruction-assumption）、`src/morphology.py`
  （none/open/close/erode/dilate/open+close，默认 open, ellipse, 3×3,
  iter 1，禁止 5×5 起步）、`src/metrics.py` 追加 `detection_metrics`
  （星点能量保留率 + 背景噪点数，engineering-choice：能量保留率 =
  目标圆盘内前景保留的灰度能量占比；噪点数 = 8 连通前景连通域中与
  目标圆盘不相交的个数）。输出 `07_binary.png`、`08_morphology.png`、
  `09_final.png`（前景掩膜红色叠加，BGR，engineering-choice）。
  参考帧策略对照（open 3×3 形态学后）：

  | 策略 | C | 能量保留率 | 背景噪点数 | 前景像素 |
  |---|---|---|---|---|
  | mu+S | – | 0.275 | 111853 | 3787955 |
  | mu+C*S（默认） | 1.5 | 0.213 | 111267 | 3640736 |
  | mu+C*S | 3.0 | 0.201 | 109632 | 3293799 |
  | mu+C*S | 5.0 | 0.101 | 107764 | 2950852 |
  | mu+C*sigma_loc | 1.5 | 0.213 | 52797 | 618655 |
  | mu+C*sigma_loc | 3.0 | 0.101 | 29185 | 241774 |
  | mu+C*sigma_loc | 5.0 | 0.063 | 10674 | 58381 |

  结论：退化拉伸下 S 在平坦块趋近 0（T≈μloc，块内约半数像素过阈值），
  故 S 系策略前景/噪点量级很大；`mu+C*sigma_loc` C=1.5 在与默认策略
  相同的能量保留率（0.213）下噪点数减半、前景像素减至 1/6。按 plan
  规定配置默认仍为 `mu+C*S`，最终策略留待 P5 四组实验综合判定。
  噪点绝对量级（10⁴–10⁵）主要源于 ±32768 坏点经双边滤波保边残留
  与背景 σ 被压缩至 0.02/255 的叠加效应。
- **P5 SNR 评价与四组实验**（完成）：`scripts/evaluate_preprocess.py`
  （A 原图 / B 双边滤波 / C 完整 pipeline / D 滤波器对照）、
  `scripts/compare_filters.py`（D 组 mean/median/gaussian/bilateral 同空间
  支持对照，可独立运行）。输出 `outputs/p5_evaluation/metrics.csv`
  （image, method, snr_before, snr_after, snr_gain_percent, background_mean/std,
  target_mean, runtime_ms + group/target_peak/energy_retention/
  background_noise_count 判据辅助列）与 `comparison.png`（原图 + D 组四滤波
  结果五图并排，标注 SNR）、`compare_filters.csv/.png`。
  参考帧结果（SNR_before=17.61）：

  | 组 | 方法 | SNR_after | 增益 | target_mean 保留率 | 峰值保留率(参考) |
  |---|---|---|---|---|---|
  | B | bilateral | 43.37 | +146.25% | 0.983 | 0.346 |
  | C | pipeline_full | 43.37 | +146.24% | 能量保留率 0.213 | – |
  | D | mean | 60.37 | +242.74% | 0.998 | 0.145 |
  | D | median | 12.10 | −31.28% | 0.220（FAIL） | 0.035 |
  | D | gaussian | 53.58 | +204.22% | 1.000 | 0.241 |

  判据（plan P5：SNR_after > SNR_before 且目标峰值/局部能量未明显破坏；
  engineering-choice：以 target_mean 保留率 ≥ 0.8 判"局部能量未破坏"，
  单像素峰值对点源平滑天然敏感，仅作参考；overall 只就论文方法链路
  B/C 判定）：**B/C PASS**。D 组结论：mean/gaussian 的 SNR 更高但纯属
  "SNR 奖励背景平滑"（与 P2 网格搜索结论一致），其峰值涂抹远重于双边
  滤波（峰值保留率 0.145/0.241 vs 0.346）；median 直接摧毁星点（PSF
  小于 5×5 核，target_mean 跌至 0.22）。综合 SNR 增益与目标保持，双边
  滤波为四者最优，印证论文选择；146% 的 SNR 增益与论文 93.37% 同量级
  （plan 不硬性要求数值一致，ROI 与数据均不同）。
- **P5 批处理**（完成）：`run_preprocess.py --batch` 对 rst19 全部 15 帧
  跑完整流水线，每帧输出 01–09 图到 `outputs/batch/<帧名>/`，汇总
  `outputs/batch/batch_metrics.csv`。15 帧 SNR 17.45–19.50 →
  40.57–48.12（+124.7% ~ +154.6%），能量保留率 0.213–0.287，
  背景噪点数 1.11×10⁵ ± 0.5%，帧间稳定。
- **测试**（完成）：`tests/test_bilateral.py`（核定义、σr 缩放、常数图
  不变、手工小矩阵对照、翻转对称性、阶跃边保边优于高斯）、
  `tests/test_background.py`（块统计已知值、边缘残块、公式 (5) S_map、
  σglo）、`tests/test_metrics.py`（合成图 SNR 已知值、σ_b=0 异常、
  detection_metrics 合成场景），`pytest tests/` 14 项全部通过。
  注意（engineering-choice）：实测 OpenCV 5.0 的 float32 bilateralFilter
  与教科书公式存在最高约 5% 偏差（探针实验确认其颜色权重形状、翻转
  对称性与保边行为均正常，疑似快速 exp 近似/权重截断），手工对照
  测试以 5% 容差防止实现层 gross error；管线有效性由 P1/P2/P5 的
  SNR 与目标保持实测保证。

显示归一化说明（engineering-choice）：PNG 落盘用 0.5%/99.5% 百分位拉伸
（±32768 量级坏点会使全局 min-max 把星点压到不可见）；管线内数据不受影响。

## ROI

`data/roi/roi.yaml`：SNR 参考 ROI，target=(1448, 1636, r=5)，背景环 inner=10 / outer=20。

**ROI 选取说明（reconstruction-assumption）**：P0 阶段对首/末帧做帧差检查后，未发现明显的暗弱
点状动目标——frame 1 顶部 (1448,370) 附近的亮条带为卫星尾迹，仅出现于单帧，不适合做点状
SNR ROI；帧差中其余 ±6×10⁴ 量级的散点为闪烁坏点。按 implementation_plan P0 的备选方案，
选取一颗孤立的中等亮度恒星（峰值 3993，背景 21）作为 SNR 参考。若拿到目标真值坐标可直接
替换 `data/roi/roi.yaml`。
