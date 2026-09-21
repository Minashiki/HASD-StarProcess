# 《复杂星空背景下暗弱目标星图预处理复现计划书》

> 复现对象：袁博等，《复杂星空背景下的暗弱空间目标层次化检测框架》  
> 复现范围：仅实现论文 1.1“图像预处理过程”，不进入目标检测、轨迹还原等后续模块。

---

## 一、项目目标

本项目仅复现论文 HASD-StarNet 的**星图预处理部分**，目标不是实现完整目标检测算法，而是得到一个独立、可验证、后续可直接接入星点匹配或动目标检测模块的预处理程序。

最终程序输入原始单帧星图，输出至少四类结果：

1. 原始灰度星图；
2. 双边滤波去噪结果；
3. 自适应二值化/背景分割结果；
4. 形态学处理后的最终预处理结果。

另外需要自动计算论文使用的 SNR 指标，输出处理前后的 SNR 及提升比例，并保存中间结果，便于后续分析。

### 本阶段明确不实现

- 三层高斯金字塔；
- 恒星质心定位；
- 恒星背景识别；
- 双向时空差分；
- 卡尔曼轨迹预测；
- 星等约束 RANSAC；
- 帧间配准；
- 多帧轨迹提取。

论文虽然在后文把高斯金字塔称为多尺度特征提取，但它位于目标检测部分，而不是 1.1 图像预处理部分。

---

## 二、需要复现的论文算法

整个预处理 pipeline 定义为：

\[
I_0
\rightarrow
I_{\mathrm{bilateral}}
\rightarrow
I_{\mathrm{norm}}
\rightarrow
M_{\mathrm{adaptive}}
\rightarrow
M_{\mathrm{morph}}
\]

其中分别对应：

**原图 → 双边滤波 → 对比度拉伸 → 局部自适应背景建模/二值化 → 形态学处理。**

### 1. 图像读取与灰度数据保护

第一步不要直接转 `uint8`。

星图很可能是：

- 8 bit；
- 12 bit；
- 16 bit；
- FITS 浮点数据。

因此程序内部建议统一转换为：

```text
float32
```

保留原始动态范围。

需要记录：

```text
width
height
dtype
min
max
mean
std
```

原始图像保存不修改，后续任何归一化都使用副本。

这一点属于工程实现而不是论文算法，但对天文图像非常重要，否则直接压成 8 bit 会损失暗弱星点。

---

## 三、阶段一：复现自适应双边滤波

论文首先使用双边滤波同时考虑**空间距离**与**灰度差异**：

\[
I'(x,y)=
\frac{1}{W_p}
\sum_{i=-k}^{k}
\sum_{j=-k}^{k}
I(x+i,y+j)
w_s(i,j)w_r(i,j)
\]

空间权重：

\[
w_s(i,j)
=
\exp
\left(
-\frac{i^2+j^2}{2\sigma_s^2}
\right)
\]

灰度权重：

\[
w_r(i,j)
=
\exp
\left[
-\frac{
\left\|I(x+i,y+j)-I(x,y)\right\|^2
}
{2\sigma_r^2}
\right]
\]

论文的目的，是在抑制随机噪声的同时尽量避免暗弱目标边缘被普通高斯滤波抹掉。

### 论文给出的参数搜索范围

首先让 Kimi Code 实现参数可配置化：

```yaml
bilateral:
  kernel_size: 5
  sigma_space: 1.5
  sigma_range: 25
```

同时保留网格搜索：

```text
kernel size ∈ {3, 5, 7}
σs ∈ [0.5, 2.5]，step = 0.5
σr ∈ [10, 30]，step = 5
```

论文最终使用：

\[
k=5,\qquad
\sigma_s=1.5,\qquad
\sigma_r=25
\]

并以输出 SNR 最大作为参数选择原则。

### 复现时必须处理的参数歧义

论文文字称：

> 核尺寸 \(k=5\)

但公式写成：

\[
\sum_{i=-k}^{k}
\]

如果严格按照公式，`k=5` 实际对应 **11×11** 邻域；而如果“核尺寸=5”按通常图像处理定义，则应是 **5×5**。

因此代码不要直接固定解释，而要实现两个模式：

```yaml
bilateral:
  kernel_definition: "diameter"
```

和：

```yaml
bilateral:
  kernel_definition: "radius"
```

默认采用通常意义上的 **5×5** 作为主实验，再用 11×11 做一次对照。

之后根据论文 SNR 和图像视觉结果判断作者真实使用的是哪一种。

---

## 四、阶段二：对比度拉伸

论文接下来明确写到，先将灰度范围归一化到：

\[
[0,255]
\]

目的是增强暗弱目标的灰度梯度。

第一版严格使用 min-max：

\[
I_n =
255
\frac{I-I_{\min}}
{I_{\max}-I_{\min}}
\]

代码单独实现：

```python
contrast_stretch(image)
```

不要一开始就使用 CLAHE、gamma、百分位裁剪等改进方法。

当前任务是**论文复现，不是算法改进**。

后面如果极亮恒星导致 min-max 拉伸效果不好，再另开：

```text
experiments/improved_contrast/
```

做 percentile stretch、CLAHE 等实验，但不能混入 paper baseline。

---

## 五、阶段三：20×20 局部背景建模

论文将图像划分为：

\[
20\times20
\]

的局部区域，然后计算每个区域的：

\[
\mu_{\mathrm{loc}}
\]

和：

\[
\sigma_{\mathrm{loc}}
\]

同时计算整幅图像的：

\[
\sigma_{\mathrm{glo}}
\]

建立局部背景模型。

论文给出的量为：

\[
S(x,y)
=
\left[
1-
\exp
\left(
-\frac{\sigma_{\mathrm{loc}}^2}
{\sigma_{\mathrm{glo}}^2}
\right)
\right]
\sigma_{\mathrm{glo}}
\]

因此这一模块先实现：

```python
local_background_statistics(
    image,
    block_size=(20, 20)
)
```

输出至少包括：

```text
local_mean_map
local_std_map
global_std
S_map
```

并把这些图全部保存出来。

建议输出：

```text
outputs/p3_stretch_background/local_mean.png
outputs/p3_stretch_background/local_std.png
outputs/p3_stretch_background/adaptive_score.png
```

这样能够直观看论文背景模型对星点、亮星和背景噪声分别有什么响应。

---

## 六、阶段四：局部自适应二值化

这里是这篇论文复现中**最需要谨慎处理的地方**。

论文明确说明：

> 采用基于局部对比度自适应增强的二值化进行背景分割，动态调整局部对比度阈值。

但是正文只给出了上述 \(S(x,y)\) 的表达式，**并没有明确给出最终**

\[
I(x,y)>T(x,y)
\]

中的完整 \(T(x,y)\) 计算公式，也没有明确说明 \(S(x,y)\) 是：

- 阈值本身；
- 阈值修正量；
- 局部对比度；
- 还是用于生成阈值的中间变量。

这一点不能让 Kimi Code 自己编一个公式然后宣称“论文复现”。

因此采用**双层实现策略**。

### Paper-faithful 层

严格实现论文确定的部分：

```text
contrast stretch
↓
20×20 background blocks
↓
μloc
σloc
σglo
↓
S(x,y)
```

到这里全部属于论文明确给出的内容。

### Reconstruction 层

将最终二值化规则做成独立策略：

```python
threshold_strategy()
```

先提供若干候选，例如：

```text
T = μloc + S
T = μloc + C·S
T = μloc + C·σloc
T = μloc - C·σloc
```

但是这些必须明确标：

```text
reconstruction / assumption
```

而不能标成：

```text
paper implementation
```

之后根据论文图 2 的输出视觉效果以及 SNR 结果选择最接近的策略。

---

## 七、阶段五：形态学处理

论文图 2 明确画出了：

> 自适应二值化 → 形态学处理 → 生成图像

但正文**没有给出形态学操作的具体参数**，包括：

- 腐蚀还是膨胀；
- 开运算还是闭运算；
- kernel 形状；
- kernel 大小；
- iteration 次数。

因此同样不能伪造论文参数。

第一版设计成：

```yaml
morphology:
  enabled: true
  operation: open
  kernel: ellipse
  kernel_size: 3
  iterations: 1
```

同时支持：

```text
none
opening
closing
erode
dilate
open + close
```

重点测试：

```text
3×3 ellipse
3×3 cross
3×3 rectangle
```

因为暗弱点目标本身可能只有几个像素，所以禁止一开始使用 5×5、7×7 强形态学处理，否则很容易直接把目标删除。

这里最终应通过**星点能量保留率 + 背景噪点数**来选择参数，而不是单纯看图是否“干净”。

---

## 八、SNR 评价模块

论文的预处理评价指标为：

\[
SNR =
\frac{|m-m_b|}
{\sigma_b}
\]

其中：

- \(m\)：目标区域平均灰度；
- \(m_b\)：目标周围背景平均灰度；
- \(\sigma_b\)：目标周围背景标准差。

所以必须实现：

```python
calculate_snr(
    image,
    target_roi,
    background_roi
)
```

这里不要让程序自动猜 target。

初期采用：

```text
人工 ROI
```

例如在配置文件中给出：

```yaml
target:
  x: 213
  y: 157
  radius: 3

background:
  inner_radius: 6
  outer_radius: 15
```

这样计算结果可重复。

后续再做自动 ROI。

---

## 九、论文结果作为复现参考，而不是硬性通过标准

论文真实星图实验中报告：

| 方法 | SNR |
|---|---:|
| 原始图像 | 5.28 |
| 均值滤波 | 9.69 |
| 中值滤波 | 7.89 |
| 高斯滤波 | 9.34 |
| 论文方法 | **10.21** |

即论文方法相比原图提高：

\[
93.37\%
\]

论文据此认为该方法在抑制噪声的同时较好地保留了有效目标信息。

但是我们的数据不是论文完全相同的原始数据、ROI 也可能不同，所以不要要求：

```text
必须达到 93.37%
```

我们的判据应该是：

\[
SNR_{\mathrm{after}}>SNR_{\mathrm{before}}
\]

同时目标峰值和局部能量不能被明显破坏。

---

## 十、工程目录结构

建议让 Kimi Code 直接创建：

```text
StarImagePreprocess/
│
├── README.md
├── requirements.txt
├── config/
│   └── paper.yaml
│
├── data/
│   ├── raw/
│   └── roi/
│
├── src/
│   ├── __init__.py
│   ├── image_io.py
│   ├── bilateral_filter.py
│   ├── contrast_stretch.py
│   ├── local_background.py
│   ├── adaptive_threshold.py
│   ├── morphology.py
│   ├── metrics.py
│   ├── visualization.py
│   └── pipeline.py
│
├── scripts/
│   ├── run_preprocess.py
│   ├── grid_search_bilateral.py
│   ├── evaluate_preprocess.py
│   └── compare_filters.py
│
├── tests/
│   ├── test_bilateral.py
│   ├── test_background.py
│   └── test_metrics.py
│
└── outputs/
    ├── filtered/
    ├── binary/
    ├── morphology/
    ├── debug/
    ├── figures/
    └── metrics/
```

不要一开始做 GUI。

先把算法复现正确，再考虑界面。

---

## 十一、Kimi Code 开发顺序

整个实现建议不要一次性让 Kimi Code “把论文复现完”，而是拆成六个 checkpoint。

| 阶段 | 工作 | 验收 |
|---|---|---|
| P0 | 建项目、图像 IO、配置系统 | 能正确加载原始星图 |
| P1 | 双边滤波 | 中间结果可保存，参数可配置 |
| P2 | 网格搜索 | 自动输出不同 \(k,\sigma_s,\sigma_r\) 的 SNR |
| P3 | 对比度拉伸 + 20×20 局部背景模型 | 输出 μ/std/S map |
| P4 | 自适应二值化 + morphology | 输出 mask 与最终结果 |
| P5 | SNR 与经典滤波对照 | 自动生成结果表与对比图 |

每完成一个阶段先运行和检查，不要让 Kimi 连续改六个模块之后才第一次测试。

---

## 十二、必须设置的四组实验

最后至少做下面四组。

### A. 原始图

```text
Raw
```

作为 baseline。

### B. 双边滤波

```text
Raw → bilateral
```

判断论文滤波本身的效果。

### C. 论文完整预处理

```text
Raw
→ bilateral
→ contrast stretch
→ adaptive threshold
→ morphology
```

这是最终 pipeline。

### D. 经典滤波对照

分别运行：

```text
mean
median
gaussian
bilateral
```

然后统一计算：

```text
SNR_before
SNR_after
SNR_gain
target_peak
background_std
runtime
```

这样基本可以复现论文第 9 图的实验逻辑。

---

## 十三、最终应生成的实验产物

Kimi Code 最终不能只给出“运行成功”。

要求它自动输出：

```text
01_raw.png
02_bilateral.png
03_contrast_stretch.png
04_local_mean.png
05_local_std.png
06_S_map.png
07_binary.png
08_morphology.png
09_final.png
```

以及：

```text
metrics.csv
grid_search.csv
```

`metrics.csv` 至少包括：

```text
image
method
snr_before
snr_after
snr_gain_percent
background_mean
background_std
target_mean
runtime_ms
```

再生成一张：

```text
comparison.png
```

把：

```text
原图
均值
中值
高斯
论文预处理
```

并排显示。

---

## 十四、当前复现中最重要的三个“不确定项”

这三个问题应现在就写进 `README.md`，避免后续被忽略。

### 1. `k=5` 的定义存在歧义

论文所谓 `k=5` 到底代表：

- 5×5 核；
- 还是半径 5 的 11×11 邻域。

需要实验验证。

### 2. 式 (5) 与最终自适应二值化之间的信息不完整

式 (5) 给出了局部背景量：

\[
S(x,y)
\]

但没有明确给出它如何转换成最终二值化阈值。

这是目前最大的复现信息缺口。

所有补充规则必须标记为：

```text
reconstruction / assumption
```

不能声称为论文原始实现。

### 3. 形态学处理参数缺失

论文图 2 有形态学处理，但正文没有报告：

- morphology 类型；
- kernel 类型；
- kernel 大小；
- iterations。

因此这部分只能进行合理重建，而不能声称完全严格复现。

---

## 复现原则

整个项目需要始终区分三类内容：

```text
1. paper-confirmed
   论文明确给出的公式、参数和流程。

2. engineering-choice
   为使程序可运行而做出的工程设计，例如 float32、配置文件、目录结构。

3. reconstruction-assumption
   论文信息缺失时为复现实验所作的假设，例如具体二值化阈值组合方式和形态学参数。
```

任何 `reconstruction-assumption` 都必须在代码、配置文件和 README 中显式标记，不允许将合理猜测写成论文原始方法。
