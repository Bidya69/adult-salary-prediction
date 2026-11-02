from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src import config


def get_models():
    """The three models compared in the report."""
    return {
        "Logistic Regression": LogisticRegression(**config.LOGREG_PARAMS),
        "Decision Tree": DecisionTreeClassifier(**config.TREE_PARAMS),
        "Random Forest": RandomForestClassifier(**config.FOREST_PARAMS),
    }


def build_pipeline(preprocessor, model, use_resampling=False):
    """
    Pipeline = preprocessing -> (optional SMOTE + Tomek) -> model.
    The imblearn Pipeline applies resampling only while fitting,
    so the test data is never resampled.
    """
    steps = [("prep", preprocessor)]
    if use_resampling:
        steps.append(("resample", SMOTETomek(random_state=config.RANDOM_STATE)))
    steps.append(("clf", model))
    return Pipeline(steps)


def build_final_model(preprocessor):
    """Final model from the PPT: Random Forest + SMOTE-Tomek."""
    forest = RandomForestClassifier(**config.FOREST_PARAMS)
    return build_pipeline(preprocessor, forest, use_resampling=True)