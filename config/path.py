import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据目录
INPUT_DIR = PROJECT_ROOT / "input"
ABSCDB_DIR = PROJECT_ROOT / "abscDB"
RESULTS_DIR = PROJECT_ROOT / "results"

# 细分输入目录
PROFILES_DIR = INPUT_DIR / "profiles"
WANG_FILE = INPUT_DIR / "coefficients_from_wang.xlsx"

# 结果子目录
IMG_DIR = RESULTS_DIR / "img"
LBL_DIR = RESULTS_DIR / "lbl"
# LOGS_DIR = RESULTS_DIR / "logs"
# MODELS_DIR = RESULTS_DIR / "models"

# 确保目录存在
for directory in [INPUT_DIR, ABSCDB_DIR, RESULTS_DIR, IMG_DIR, LBL_DIR]:
    os.makedirs(directory, exist_ok=True)