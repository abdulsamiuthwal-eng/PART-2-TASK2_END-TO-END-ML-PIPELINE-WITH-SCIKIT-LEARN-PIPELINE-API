import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

from data_ingestion import load_data, download_data
from pipeline import get_preprocessing_pipeline

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Ingest Data
    csv_path = os.path.join(DATA_DIR, "Telco-Customer-Churn.csv")
    if not os.path.exists(csv_path):
        print("Dataset not found locally, downloading...")
        download_data(dest_path=csv_path)
        
    df = load_data(csv_path)
    
    # 2. Extract features and target
    # Target encoding: Yes -> 1, No -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]
    
    # 3. Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Save the test set for decoupled evaluation
    X_test.to_csv(os.path.join(DATA_DIR, "test_features.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, "test_target.csv"), index=False)
    print("Saved test splits to data/ for evaluation.")
    
    # 4. Get the preprocessing pipeline
    preprocessor = get_preprocessing_pipeline()
    
    # 5. Define pipelines for each model
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    
    # 6. Define parameter grids for GridSearchCV
    # We prefix param names with 'classifier__' to target the classifier step in the pipeline
    lr_param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs", "saga"],
        "classifier__penalty": ["l2"]
    }
    
    rf_param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5, 10]
    }
    
    # 7. GridSearchCV for Logistic Regression
    print("\n--- Tuning Logistic Regression ---")
    lr_grid = GridSearchCV(
        lr_pipeline, lr_param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
    )
    lr_grid.fit(X_train, y_train)
    print(f"Logistic Regression Best Params: {lr_grid.best_params_}")
    print(f"Logistic Regression Best CV F1 Score: {lr_grid.best_score_:.4f}")
    
    # 8. GridSearchCV for Random Forest
    print("\n--- Tuning Random Forest ---")
    rf_grid = GridSearchCV(
        rf_pipeline, rf_param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
    )
    rf_grid.fit(X_train, y_train)
    print(f"Random Forest Best Params: {rf_grid.best_params_}")
    print(f"Random Forest Best CV F1 Score: {rf_grid.best_score_:.4f}")
    
    # 9. Compare and select the best model
    # We will evaluate both best estimators on the validation/test set to choose the absolute best
    # or compare their best CV score. Here we use the F1-score on the training CV fold.
    lr_cv_score = lr_grid.best_score_
    rf_cv_score = rf_grid.best_score_
    
    if lr_cv_score >= rf_cv_score:
        print(f"\nLogistic Regression wins with CV F1 score: {lr_cv_score:.4f} (RF: {rf_cv_score:.4f})")
        best_pipeline = lr_grid.best_estimator_
        model_name = "Logistic Regression"
    else:
        print(f"\nRandom Forest wins with CV F1 score: {rf_cv_score:.4f} (LR: {lr_cv_score:.4f})")
        best_pipeline = rf_grid.best_estimator_
        model_name = "Random Forest"
        
    # 10. Export the best model pipeline using joblib
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nExported the best pipeline ({model_name}) to: {MODEL_PATH}")
    
if __name__ == "__main__":
    main()
