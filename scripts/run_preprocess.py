"""预处理流水线入口脚本（P1–P4 全流程：读图 → 双边滤波 → 对比度拉伸
→ 局部背景建模 → 自适应二值化 → 形态学，输出 01–09 系列图）。

用法：
    conda run -n HASD-StarNet python scripts/run_preprocess.py --config config/paper.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import main

if __name__ == "__main__":
    main()
