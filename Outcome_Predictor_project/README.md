# Ensemble Learning: Student Performance Prediction

This project applies a range of ensemble learning techniques — bagging, boosting, voting, and stacking — to predict student outcomes from an online learning platform, covering both a **classification task** (course completion) and a **regression task** (final score).

## Files

| File | Description |
|---|---|
| `Ensemble_Learning_Student_Performance_Prediction.ipynb` | Main notebook: data prep, model training, evaluation, and final report |
| `Practical_Exam_Theory_Answers.ipynb` | Theory answers covering regression, regularization, overfitting, and cross-validation |
| `students_dataset.csv` | Raw dataset (5,200 rows × 19 columns) |

## Dataset

`students_dataset.csv` contains 5,200 student records with 19 columns, including:

- **Demographics / context:** `age`, `country_region`, `device_type`, `education_background`
- **Course info:** `course_level`, `course_category`, `course_start_date`, `week_of_year`
- **Engagement metrics:** `sessions`, `time_spent_hours`, `videos_watched`, `quiz_attempts`, `assignments_submitted`, `forum_posts`, `attendance_rate`
- **Targets:**
  - `completion_status` (binary) → classification target
  - `final_score` (continuous) → regression target

## Workflow

**Part B — Dataset Understanding & Preparation**
Feature/target identification, missing value handling, encoding, and an 80/20 train-test split.

**Part C — Bagging**
Bagging Classifier and Regressor built on decision tree base estimators, compared against a single base model.

**Part D — Boosting**
AdaBoost, Gradient Boosting, LightGBM, and XGBoost classifiers/regressors, with analysis of learning rate, number of estimators, and training efficiency.

**Part E — Voting & Stacking**
Hard vs. soft voting ensembles, and stacking classifiers/regressors with a meta-learner.

**Part F — Model Evaluation & Comparison**
Classification models scored on accuracy, precision, recall, F1, and ROC-AUC; regression models scored on MAE, RMSE, and R².

**Part G — Final Analysis & Reporting**
Written comparison of bagging vs. boosting, and an overall project conclusion.

## Key Results

- **Best regression model — Bagging Regressor**: MAE ≈ 0.0071, RMSE ≈ 0.0322, R² ≈ 0.999994
- **Best classification model — Stacking Classifier** (recommended): combines multiple base learners for the most robust, generalized performance, edging out the individual models and the voting ensemble.
- Across most classifiers, accuracy/precision/recall/F1 were near-perfect, suggesting the engagement features are highly predictive of course completion.

## Requirements

```
pandas
numpy
scikit-learn
lightgbm
xgboost
matplotlib / seaborn (for plots)
```

## How to Run

1. Place `students_dataset.csv` in the same directory as the notebook.
2. Open `Ensemble_Learning_Student_Performance_Prediction.ipynb` in Jupyter.
3. Run all cells top to bottom (Part B → Part G).
