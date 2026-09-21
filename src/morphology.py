"""形态学处理模块（P4）。

reconstruction-assumption: 论文仅在图 2 中出现形态学处理，未给出任何参数。
默认 open, ellipse, 3x3, iter 1；implementation_plan 明确禁止 5x5 起步
（过大的结构元会抹掉暗弱点状目标）。

支持操作（op）：
  none       原样返回
  open       开运算（去孤立噪点）
  close      闭运算（填小空洞）
  erode      腐蚀
  dilate     膨胀
  open+close 先开后闭

结构元形状（kernel_shape）：ellipse / rect / cross，边长 kernel_size（奇数为宜，
engineering-choice: 不强制校验奇偶，由 cv2.getStructuringElement 处理）。
"""

import cv2
import numpy as np

OPS = ("none", "open", "close", "erode", "dilate", "open+close")

_SHAPES = {
    "ellipse": cv2.MORPH_ELLIPSE,
    "rect": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
}


def apply_morphology(binary, op="open", kernel_shape="ellipse",
                     kernel_size=3, iterations=1):
    """对 uint8 二值图执行形态学操作，返回 uint8 二值图。"""
    img = np.asarray(binary)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if op not in OPS:
        raise ValueError(f"未知形态学操作 {op!r}，候选: {', '.join(OPS)}")
    if op == "none":
        return img
    if kernel_shape not in _SHAPES:
        raise ValueError(
            f"未知结构元形状 {kernel_shape!r}，候选: {', '.join(_SHAPES)}"
        )
    k = int(kernel_size)
    if k <= 0:
        raise ValueError("kernel_size 必须为正整数")
    kernel = cv2.getStructuringElement(_SHAPES[kernel_shape], (k, k))
    it = int(iterations)

    if op == "open":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=it)
    if op == "close":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=it)
    if op == "erode":
        return cv2.erode(img, kernel, iterations=it)
    if op == "dilate":
        return cv2.dilate(img, kernel, iterations=it)
    # open+close: 先开运算去噪点，再闭运算填目标内部小空洞
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=it)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=it)
