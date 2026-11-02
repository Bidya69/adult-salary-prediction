# Census Income Prediction

Predicts whether a person's annual income exceeds $50,000 using the Adult Income dataset. The project compares Logistic Regression, Decision Tree and Random Forest, and handles class imbalance with SMOTE + Tomek Links. The final model is a Random Forest trained on the resampled data.

## Problem

Only about 24% of people in the dataset earn more than $50K. Models trained on such imbalanced data tend to favour the majority class and miss many high earners, which shows up as low recall for the `>50K` group.

## Approach

1. Load and clean the data (strip spaces, treat `?` as missing, encode the target as 0/1)
2. Stratified 80/20 train-test split, done before any resampling to avoid data leakage
3. Preprocessing: median/most-frequent imputation, one-hot encoding, standard scaling
4. Train Logistic Regression, Decision Tree and Random Forest, each with and without SMOTE + Tomek Links
5. Final model: Random Forest + SMOTE-Tomek
6. Evaluate with Accuracy, Precision, Recall, F1, a confusion matrix and 5-fold cross-validation
7. Plot feature importance

## Project structure

```
census-income-prediction/
├── data/adult.csv
├── src/
│   ├── config.py          settings and paths
│   ├── data_loader.py     load, clean, explore, split
│   ├── preprocessing.py   imputing, encoding, scaling
│   ├── models.py          models and pipelines
│   └── evaluate.py        metrics, plots, cross-validation
├── outputs/               generated plots and results
├── main.py                runs the full pipeline
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
python main.py
```

Place the dataset at `data/adult.csv` first. The Adult Income dataset is available on Kaggle and the UCI Machine Learning Repository.

## Outputs

Saved in `outputs/`:
- `class_distribution.png`
- `model_comparison.csv`
- `confusion_matrix.png`
- `feature_importance.png`

## Results

Replace this section with the numbers from your own run on the real dataset:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (no resampling) | | | | |
| Random Forest (no resampling) | | | | |
| Random Forest + SMOTE-Tomek | | | | |

## Tech stack

Python, Pandas, NumPy, Scikit-learn, imbalanced-learn, Matplotlib, Seaborn

## Team

Group 5: Vedant Pradhan, Bidya Prasad Sahoo, Amrit Pritam Mohanty, Ashutosh Sahoo, Arpit Sarangi