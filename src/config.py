from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "adult.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# ---------- Data ----------
TARGET_COL = "income"
POSITIVE_CLASS = ">50K"          # the high-income (minority) class
MISSING_MARKER = "?"             # how missing values appear in the CSV

# ---------- Train / test split ----------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------- Cross-validation ----------
CV_FOLDS = 5

# ---------- Model settings ----------
LOGREG_PARAMS = {"max_iter": 1000}
TREE_PARAMS = {"max_depth": 10, "random_state": RANDOM_STATE}
FOREST_PARAMS = {
    "n_estimators": 300,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}