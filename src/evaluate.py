import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import config


def evaluate_model(name, pipe, X_train, X_test, y_train, y_test):
    """Fit one pipeline and return its metrics on the test set."""
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    }


def save_comparison(results):
    """Turn the list of result dicts into a table and save it as CSV."""
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    results_df = pd.DataFrame(results).round(3)
    results_df.to_csv(config.OUTPUT_DIR / "model_comparison.csv", index=False)
    return results_df


def final_report(model, X_test, y_test):
    """Classification report and confusion matrix for the final model."""
    pred = model.predict(X_test)
    print(classification_report(y_test, pred, target_names=["<=50K", ">50K"]))

    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")
    plt.savefig(config.OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def cross_validate(model, X, y):
    """5-fold CV on F1. Resampling happens only inside the training folds."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
    print(f"{config.CV_FOLDS}-fold CV F1 (>50K class): {scores.mean():.3f} +/- {scores.std():.3f}")
    return scores


def plot_feature_importance(model, num_cols, cat_cols, top_n=15):
    """Bar chart of the most important features in the Random Forest."""
    forest = model.named_steps["clf"]
    encoder = model.named_steps["prep"].named_transformers_["cat"].named_steps["encoder"]
    feature_names = num_cols + encoder.get_feature_names_out(cat_cols).tolist()

    importance = (pd.Series(forest.feature_importances_, index=feature_names)
                    .sort_values(ascending=False).head(top_n))

    importance.sort_values().plot(kind="barh", figsize=(8, 6),
                                  title=f"Top {top_n} feature importances")
    plt.savefig(config.OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    return importance