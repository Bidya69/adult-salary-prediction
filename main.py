import pandas as pd

from src import config
from src.data_loader import load_data, explore, split_data
from src.preprocessing import build_preprocessor
from src.models import get_models, build_pipeline, build_final_model
from src.evaluate import (evaluate_model, save_comparison, final_report,
                          cross_validate, plot_feature_importance)


def main():
    # 1. Load, clean and explore the data
    print("=" * 60, "\nSTEP 1: Load and explore data\n" + "=" * 60)
    df = load_data()
    explore(df)

    # 2. Split into train and test sets
    X, y, X_train, X_test, y_train, y_test = split_data(df)
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # 3. Build the preprocessor (missing values, encoding, scaling)
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # 4. Compare all models, without and with SMOTE-Tomek
    print("\n" + "=" * 60, "\nSTEP 2: Compare models\n" + "=" * 60)
    results = []
    for use_resampling in (False, True):
        suffix = "+ SMOTE-Tomek" if use_resampling else "(no resampling)"
        for name, model in get_models().items():
            pipe = build_pipeline(preprocessor, model, use_resampling)
            results.append(
                evaluate_model(f"{name} {suffix}", pipe,
                               X_train, X_test, y_train, y_test)
            )
            print(f"Done: {name} {suffix}")

    results_df = save_comparison(results)
    print("\n", results_df.to_string(index=False))

    # 5. Final model: Random Forest + SMOTE-Tomek
    print("\n" + "=" * 60, "\nSTEP 3: Final model (Random Forest + SMOTE-Tomek)\n" + "=" * 60)
    final_model = build_final_model(preprocessor)
    final_model.fit(X_train, y_train)
    final_report(final_model, X_test, y_test)

    # 6. Cross-validation
    print("\n" + "=" * 60, "\nSTEP 4: Cross-validation\n" + "=" * 60)
    cross_validate(build_final_model(preprocessor), X, y)

    # 7. Feature importance
    print("\n" + "=" * 60, "\nSTEP 5: Feature importance\n" + "=" * 60)
    importance = plot_feature_importance(final_model, num_cols, cat_cols)
    print(importance.round(3))

    # 8. Predict for one person from the test set
    print("\n" + "=" * 60, "\nSTEP 6: Sample prediction\n" + "=" * 60)
    sample = X_test.iloc[[0]]
    prediction = final_model.predict(sample)[0]
    probability = final_model.predict_proba(sample)[0][1]
    print(sample.T)
    print(f"\nPredicted: {'>50K' if prediction == 1 else '<=50K'} "
          f"(probability of >50K: {probability:.2f})")
    print(f"Actual:    {'>50K' if y_test.iloc[0] == 1 else '<=50K'}")

    print(f"\nAll plots and results saved in: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()