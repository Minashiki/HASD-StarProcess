# 实施计划：星图预处理复现（HASD-StarNet preprocessing）

> 依据 `plan.md`（复现计划书）与 `ref/A复杂星空背景下的暗弱空间目标层次化检测框架_袁博.pdf`（已提取文本核实）制定。本文件仅记录实施方案，暂不实施。

## 一、背景核实结论

### 论文核实（PyMuPDF 提取 PDF 文本确认）

- 双边滤波公式 (1)–(4) 与 plan.md 一致；参数 k=5, σs=1.5, σr=25；网格搜索范围 k∈{3,5,7}, σs∈[0.5,2.5] step 0.5, σr∈[10,30] step 5；SNR 提升 93.37%。
- S(x,y) = [1−exp(−σloc²/σglo²)]·σglo（公式 5）；μloc 只在文字中出现，未进公式。
- **缺失项确认**：1.1 节无显式 T(x,y) 二值化判决式；形态学仅在图 2 出现，无参数。
- 可作重建参考的相邻公式：1.2.1 节 T = Wμ(x,y) − C·Wσ(x,y)（C=1.5, T′=T+30），属恒星背景分割而非预处理，引用时必须标记 `reconstruction-assumption`。
- SNR = |m−m_b|/σ_b（线性比值，论文称 dB）；真实星图数值 5.28/9.69/7.89/9.34/10.21 均属实。

### 数据核实（`rst19/`）

- 15 帧 FITS 序列 + 1 份格式说明 docx。
- FITS 头：BITPIX=16（BZERO=0, BSCALE=1），4096×4096，曝光 1500 ms，IMAGETYP=OBJECT。相机数据实为 uint16，读取时将 int16 按位重解释为 uint16。
- 读取用 `astropy.io.fits`，内部统一转 float32。

### 环境核实

- conda 环境 `HASD-StarNet`（Python 3.12.14）目前仅 6 个包，需安装依赖。
- 所有 python/pip 调用使用 `conda run -n HASD-StarNet ...`。

### 项目根

- `D:\StarImgPro02` 本身就是项目根（.gitignore / requirements.txt / README.md 放根目录），不再建 `StarImagePreprocess/` 子目录。

## 二、目录结构

```text
.gitignore            # 忽略 outputs/、rst19/（体积大）、__pycache__/、.pytest_cache/ 等
requirements.txt
README.md             # 含三个不确定项 + paper-confirmed / engineering-choice / reconstruction-assumption 三分原则
config/
  paper.yaml          # 论文参数（默认配置）
data/roi/             # ROI 配置 yaml
src/
  __init__.py
  image_io.py           # FITS 读取 → float32，记录 width/height/dtype/min/max/mean/std
  bilateral_filter.py   # 自适应双边滤波，kernel_definition: diameter | radius
  contrast_stretch.py   # min-max → [0,255]
  local_background.py   # 20×20 块统计：local_mean_map/local_std_map/global_std/S_map
  adaptive_threshold.py # threshold_strategy()：多候选，全部标记 reconstruction
  morphology.py         # none/open/close/erode/dilate/open+close，默认 3×3 ellipse
  metrics.py            # calculate_snr(image, target_roi, background_roi)
  visualization.py      # 保存中间图、comparison.png
  pipeline.py           # 串联全流程，按帧输出 01–09 系列图
scripts/
  run_preprocess.py
  grid_search_bilateral.py
  evaluate_preprocess.py
  compare_filters.py    # mean/median/gaussian/bilateral 对照
tests/
  test_bilateral.py
  test_background.py
  test_metrics.py
outputs/                # 按阶段命名：p0_debug/ p1_bilateral/ p2_grid_search/ p3_stretch_background/ p4_binary_morphology/（gitignore）
```

## 三、依赖（requirements.txt）

`numpy, astropy, opencv-python, matplotlib, pandas, pyyaml, pytest`

双边滤波第一版用 OpenCV `cv2.bilateralFilter`（与论文公式等价），diameter/radius 两种核定义模式均支持。

## 四、执行步骤（按 plan.md 的 P0–P5 checkpoint，每步跑通验证后再进下一步）

### P0 环境与工程

- 安装依赖；建目录、`.gitignore`、`requirements.txt`、`README.md`（含三个不确定项与三分原则）、`config/paper.yaml`。
- `src/image_io.py`：读取 FITS（astropy，int16→uint16 按位重解释 → float32），记录 stats（dtype=uint16, 4096×4096, min/max/mean/std）。
- 验收：能加载一帧并输出统计。
- 可视化检查一帧，人工选取一个暗弱目标 ROI 写入 `data/roi/roi.yaml`（target x/y/radius + 背景环 inner/outer radius），供 SNR 计算；若无明显动目标，则选一颗中等亮度恒星作 SNR 参考 ROI，并在 README 注明。

### P1 双边滤波（bilateral_filter.py + pipeline.py 骨架）

- 配置：`kernel_size: 5, sigma_space: 1.5, sigma_range: 25, kernel_definition: diameter`（5×5 主实验；radius 模式即 11×11 对照）。
- 注意 σr=25 是 [0,255] 尺度下的值，而论文流程中滤波在拉伸之前——float 原始数据需按动态范围缩放 σr（可配置），此属 engineering-choice，需在代码和 README 记录。
- 验收：输出 `outputs/p1_bilateral/02_bilateral.png`，打印 SNR(before/after)。

### P2 网格搜索（scripts/grid_search_bilateral.py）

- k∈{3,5,7} × σs∈{0.5..2.5} × σr∈{10..30}（diameter 模式），逐组合算 SNR，写 `outputs/p2_grid_search/grid_search.csv`，取 SNR 最大组合与论文 (5,1.5,25) 对照。
- 另跑 radius（11×11）模式关键组合作对照，辅助判断论文 k=5 的真实含义。
- 4096×4096 大图 × 45 组合较慢：以单帧跑，必要时裁剪 ROI 加速（engineering-choice，写明）。

### P3 对比度拉伸 + 局部背景模型

- min-max → [0,255]（paper-faithful，不用 CLAHE/percentile）。
- 20×20 块：μloc/σloc 图、σglo、S_map；存 `outputs/p3_stretch_background/04_local_mean.png, 05_local_std.png, 06_S_map.png`。

### P4 自适应二值化 + 形态学

- 候选策略（全部标 `reconstruction-assumption`）：`T = μloc + S`、`T = μloc + C·S`、`T = μloc + C·σloc`（对照论文 1.2.1 式，C=1.5）。默认 `μloc + C·S`，C 可配。
- 形态学默认 `open, ellipse, 3×3, iter 1`；禁止 5×5 起步。
- 按"星点能量保留率 + 背景噪点数"选参数，输出 `07_binary.png, 08_morphology.png, 09_final.png`。

### P5 SNR 评价与四组实验

- A 原图 / B 双边滤波 / C 完整 pipeline / D mean·median·gaussian·bilateral 对照。
- 输出 `metrics.csv`（image, method, snr_before, snr_after, snr_gain_percent, background_mean/std, target_mean, runtime_ms）与 `comparison.png`（五图并排）。
- 判据：SNR_after > SNR_before 且目标峰值/局部能量未明显破坏；不硬性要求 93.37%。
- 单帧（参考帧）跑通全部实验后，用 `run_preprocess.py` 对 rst19 全部 15 帧批处理输出 01–09 图。

### 测试

- `tests/`：bilateral 权重对称性/与手工小矩阵对照、local_background 块统计正确性、metrics 合成图像 SNR 已知值。
- `conda run -n HASD-StarNet python -m pytest tests/` 通过。

## 五、复现纪律（贯穿全程）

- 代码注释与 README 显式区分 `paper-confirmed` / `engineering-choice` / `reconstruction-assumption`。
- 不做 GUI；不做高斯金字塔、质心、配准等检测模块。
- 每 checkpoint 先运行检查再进入下一个。

## 六、主要风险

- 4096×4096 × 15 帧 × 网格搜索耗时：以单帧为主、批处理放最后，必要时裁剪 ROI。
- σr=25 的灰度尺度歧义（论文隐含 [0,255]）：实现可配置并记录假设。
- ROI 需人工选：P0 阶段看图后定坐标；若后续拿到目标真值坐标可直接替换 `data/roi/roi.yaml`。
