# Task 2: End-to-End ML Pipeline with Scikit-Learn

## Objective
Predict customer churn using a production-ready ML pipeline built with Scikit-Learn's `Pipeline` and `ColumnTransformer` APIs.

---

## Dataset
**Telco Customer Churn Dataset**  
Source: [Kaggle / GitHub Mirror](https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv)  
- 7,043 customer records  
- 20 features (demographic, service, billing info)  
- Target: `Churn` (Yes/No)

---

## Methodology

1. **Data Ingestion** — Dataset downloaded programmatically via `data_ingestion.py`
2. **Preprocessing Pipeline** — Custom transformers for:
   - Cleaning `TotalCharges` blank strings → `NaN` → imputed with median
   - Scaling numerical features with `StandardScaler`
   - Encoding categorical features with `OneHotEncoder`
3. **Model Training** — Two models trained with `GridSearchCV` (5-fold CV):
   - Logistic Regression
   - Random Forest
4. **Best Model Export** — Saved via `joblib` to `models/best_model.joblib`
5. **Evaluation** — Metrics and plots generated on holdout test set

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.8055 |
| Precision | 0.6572 |
| Recall | 0.5588 |
| F1 Score | 0.6040 |
| ROC-AUC | 0.8411 |

**Best Model:** Logistic Regression (CV F1: 0.5994 vs Random Forest: 0.5810)

---

## Project Structure

```
├── data/                  # Downloaded dataset
├── models/                # Saved best model (.joblib)
├── notebooks/             # Jupyter Notebook (EDA + full pipeline)
├── reports/               # Confusion Matrix & ROC Curve plots
├── src/
│   ├── data_ingestion.py  # Dataset download
│   ├── pipeline.py        # Preprocessing pipeline
│   ├── train.py           # Training & hyperparameter tuning
│   └── evaluate.py        # Model evaluation
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline
python src/data_ingestion.py
python src/train.py
python src/evaluate.py
```

---

## Key Observations

- Logistic Regression outperformed Random Forest on this dataset
- ROC-AUC of **0.84** shows strong discriminative ability
- Class imbalance (churn ~26%) caused lower F1 on minority class — expected behavior
