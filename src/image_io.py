"""FITS 星图读取模块。

paper-confirmed: rst19 数据为 BITPIX=16（BZERO=0, BSCALE=1）、
4096x4096、曝光 1500 ms、IMAGETYP=OBJECT。
data-corrected: 相机实际输出为无符号 uint16，FITS 按 int16 存储导致负值；
读取时对 int16 做按位重解释为 uint16（-32768→32768, -1→65535）。
engineering-choice: 读取后内部统一转 float32，后续所有处理在 float32 上进行。
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits


@dataclass
class ImageStats:
    """单帧图像的基本统计信息。"""

    width: int
    height: int
    dtype: str          # 重解释后的数据类型（如 uint16）
    min: float
    max: float
    mean: float
    std: float
    header: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"size={self.width}x{self.height}, dtype={self.dtype}, "
            f"min={self.min:.1f}, max={self.max:.1f}, "
            f"mean={self.mean:.2f}, std={self.std:.2f}"
        )


def _safe_header_dict(header):
    """逐卡提取 header，跳过无法解析的卡（rst19 的 AZIMUTH 卡不符合 FITS 规范）。

    engineering-choice: 畸形卡仅影响元数据，不影响图像数据，静默跳过。
    """
    out = {}
    for card in header.cards:
        try:
            out[card.keyword] = card.value
        except Exception:
            continue
    return out


def load_fits(path):
    """读取一帧 FITS，返回 (float32 图像, ImageStats)。

    data-corrected: BITPIX=16 且 BZERO=0 时，将 int16 按位重解释为 uint16
    （相机原始数据为无符号），消除因错误按有符号读取产生的负值。
    engineering-choice: astropy 按 BZERO/BSCALE 应用缩放后数据即为物理值；
    这里直接取 primary HDU 数据并转 float32。
    engineering-choice: rst19 文件缺少 END 填充块（截断警告），
    用 ignore_missing_end=True 容忍。
    """
    with fits.open(path, ignore_missing_end=True) as hdul:
        raw = hdul[0].data
        header = _safe_header_dict(hdul[0].header)
    if raw is None:
        raise ValueError(f"FITS 文件无图像数据: {path}")

    if raw.dtype.kind == "i" and raw.dtype.itemsize == 2 and not header.get("BZERO"):
        raw = raw.view(raw.dtype.str.replace("i", "u"))

    image = np.asarray(raw, dtype=np.float32)
    stats = ImageStats(
        width=image.shape[1],
        height=image.shape[0],
        dtype=str(raw.dtype),
        min=float(image.min()),
        max=float(image.max()),
        mean=float(image.mean()),
        std=float(image.std()),
        header=header,
    )
    return image, stats


def save_fits(path, image):
    """把管线中的数组原样写入 FITS（自动建目录）。返回路径字符串。

    engineering-choice: dtype 与数值完全保留（float32 中间结果写 BITPIX=-32，
    uint8 二值图写 BITPIX=8），不做任何归一化/拉伸，供后续直接用数值复核；
    与 save_png 的显示归一化互不影响。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, np.asarray(image), overwrite=True)
    return str(path)


if __name__ == "__main__":
    import sys

    img, st = load_fits(sys.argv[1])
    print(st)
    print(f"IMAGETYP={st.header.get('IMAGETYP')}, EXPTIME={st.header.get('EXPTIME')}")
