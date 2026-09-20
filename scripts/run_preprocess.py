"""预处理流水线入口脚本（当前 P1：读图 → 双边滤波 → SNR）。

用法：
    conda run -n HASD-StarNet python scripts/run_preprocess.py --config config/paper.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import main

if __name__ == "__main__":
    main()
