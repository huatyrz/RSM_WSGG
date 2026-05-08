import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据目录
INPUT_DIR = PROJECT_ROOT / "input"
ABSCDB_DIR = PROJECT_ROOT / "abscDB"
RESULTS_DIR = PROJECT_ROOT / "results"

# 结果子目录
IMG_DIR = RESULTS_DIR / "img"
# LOGS_DIR = RESULTS_DIR / "logs"
# MODELS_DIR = RESULTS_DIR / "models"

# 确保目录存在
for directory in [INPUT_DIR, ABSCDB_DIR, RESULTS_DIR, IMG_DIR]:
    os.makedirs(directory, exist_ok=True)