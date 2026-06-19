import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

# Identify numerical and categorical columns
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

def parse_total_charges(X):
    """Converts a column (typically TotalCharges) to float, replacing empty spaces with NaN.
    
    Args:
        X: 2D array or DataFrame containing the column to parse.
        
    Returns:
        A DataFrame with the parsed column.
    """
    # Convert input to DataFrame if it's a numpy array
    df = pd.DataFrame(X).copy()
    for col in df.columns:
        # Cast to string, strip whitespace, replace empty strings with NaN, and convert to numeric
        df[col] = pd.to_numeric(df[col].astype(str).str.strip().replace("", np.nan), errors="coerce")
    return df

def get_preprocessing_pipeline() -> ColumnTransformer:
    """Creates the Scikit-Learn ColumnTransformer for preprocessing the Telco Churn dataset.

    Returns:
        A fitted or unfitted ColumnTransformer.
    """
    # Pipeline for TotalCharges (which needs string-to-float parsing, imputation, and scaling)
    total_charges_pipeline = Pipeline(steps=[
        ("parser", FunctionTransformer(parse_total_charges, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Standard numerical pipeline (imputation and scaling)
    standard_num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Preprocessor for numerical columns
    # We apply total_charges_pipeline to TotalCharges, and standard_num_pipeline to others
    # Note: ColumnTransformer requires column selections. Let's split them.
    # We can also just run the parsing step on all numerical features (it's a no-op on float columns)
    # to keep it simple, but separating is cleaner and more explicit.
    
    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        # Cast SeniorCitizen to string to treat it as categorical, or simple imputer
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Unified preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("tc_num", total_charges_pipeline, ["TotalCharges"]),
            ("std_num", standard_num_pipeline, ["tenure", "MonthlyCharges"]),
            ("cat", categorical_pipeline, CATEGORICAL_COLS)
        ],
        remainder="drop"
    )
    
    return preprocessor
