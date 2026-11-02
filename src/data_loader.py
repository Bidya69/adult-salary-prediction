import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src import config


def load_data():
    """Read the CSV and do basic cleaning."""
    df = pd.read_csv(config.DATA_PATH)

    # "marital.status" -> "marital_status", so all column names are consistent
    df.columns = [c.strip().replace(".", "_").replace("-", "_") for c in df.columns]

    # Remove stray spaces in text columns and mark "?" as missing
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df = df.replace(config.MISSING_MARKER, np.nan)

    # Target: 1 if income > 50K, otherwise 0 (also handles values like "<=50K.")
    df[config.TARGET_COL] = df[config.TARGET_COL].str.replace(".", "", regex=False)
    df[config.TARGET_COL] = (df[config.TARGET_COL] == config.POSITIVE_CLASS).astype(int)

    return df


def explore(df):
    """Print basic info and save the class distribution plot."""
    print("Shape:", df.shape)

    missing = df.isna().sum()
    print("\nMissing values:\n", missing[missing > 0])

    dist = df[config.TARGET_COL].value_counts(normalize=True) * 100
    print("\nClass distribution (%):\n", dist.round(1))

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    sns.countplot(x=config.TARGET_COL, data=df)
    plt.title("Class distribution (0 = <=50K, 1 = >50K)")
    plt.savefig(config.OUTPUT_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def split_data(df):
    """Split BEFORE any resampling so the test set stays untouched."""
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,                     # keeps the 76:24 ratio in both sets
        random_state=config.RANDOM_STATE,
    )
    return X, y, X_train, X_test, y_train, y_test