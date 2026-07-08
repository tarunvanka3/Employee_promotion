# Employee Promotion Prediction

A machine learning project that predicts whether an employee is likely to be promoted, based on HR data such as department, education, performance ratings, training scores, and tenure. The project covers the full pipeline — data cleaning, preprocessing, exploratory analysis, handling class imbalance, and training/comparing multiple classification models.

## Overview

Employee promotion decisions are often manual and time-consuming for large organizations. This project builds a classification model that uses historical employee data to predict the `is_promoted` outcome (0 = not promoted, 1 = promoted), helping HR teams identify likely promotion candidates more objectively.

## Dataset

The project uses a **train** and **test** HR dataset (`train.csv`, `test.csv`) with employee-level features, including:

- `employee_id`, `department`, `region`, `education`, `gender`
- `recruitment_channel`, `no_of_trainings`, `age`
- `previous_year_rating`, `length_of_service`
- `KPIs_met >80%`, `awards_won?`, `avg_training_score`
- `is_promoted` (target variable)

> **Note:** The dataset paths in the notebook are set to a local Windows directory (`C:\Users\gowth\...`). Update these paths to your own local file locations before running the notebook.

## Project Workflow

1. **Data Loading** — Import train and test CSV files, keep backup copies.
2. **Data Exploration** — Inspect shape, data types, null values, duplicates, and unique value counts.
3. **Missing Value Treatment**
   - `previous_year_rating` (numeric) → imputed using **KNN Imputer**
   - `education` (categorical) → imputed using **Simple Imputer** (most frequent strategy)
4. **Categorical Encoding** — `department`, `region`, `education`, `gender`, and `recruitment_channel` encoded using **Label Encoder**.
5. **Class Imbalance Check** — Analyze the proportion of promoted vs. non-promoted employees.
6. **Feature Selection** — Drop non-predictive columns (`employee_id`, `gender`, `region`).
7. **Handling Imbalanced Data** — Apply **Random Oversampling** on the minority class.
8. **Train-Test Split** — 70/30 split using `train_test_split`.
9. **Feature Scaling** — Normalize features using **MinMaxScaler**.
10. **Exploratory Data Analysis** — Visualize target distribution and promotion rate by education using Seaborn.
11. **Model Building** — Train and evaluate multiple classification algorithms:
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine — Linear, Polynomial, Gaussian (RBF), and Sigmoid kernels
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Extra Trees Classifier
12. **Model Evaluation** — Compare models using confusion matrix, classification report (precision, recall, F1-score), and accuracy.
13. **Model Comparison** — Consolidate results across all algorithms to identify the best-performing model.
14. **Prediction on Test Data** — Generate promotion predictions for the test dataset and merge them with original records.
15. **Visualization** — Plot the trained decision tree for interpretability.

## Tech Stack

| Category | Tools/Libraries |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy, PandasSQL |
| Visualization | Matplotlib, Seaborn |
| Preprocessing | Scikit-learn (`KNNImputer`, `SimpleImputer`, `LabelEncoder`, `MinMaxScaler`) |
| Imbalanced Data | imbalanced-learn (`RandomOverSampler`) |
| Modeling | Scikit-learn (KNN, SVM, Logistic Regression, Decision Tree, Random Forest, Extra Trees) |

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn pandasql
```

## Usage

1. Clone this repository.
2. Place `train.csv` and `test.csv` in a local folder and update the file paths in the notebook.
3. Open the notebook:
   ```bash
   jupyter notebook Employee_Promotion__Analysis.ipynb
   ```
4. Run the cells sequentially to reproduce data preprocessing, model training, and evaluation.

## Results

Multiple classification algorithms are trained and compared on the same preprocessed dataset. Evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix) are generated for each model to identify the best-performing algorithm for predicting employee promotions. The final model's predictions are merged back with the test dataset for review.

## Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for top-performing models
- Feature importance analysis to identify key promotion drivers
- Cross-validation for more robust performance estimates
- Deployment as a simple API or web app for HR use

## Author

Tarun Vanka
