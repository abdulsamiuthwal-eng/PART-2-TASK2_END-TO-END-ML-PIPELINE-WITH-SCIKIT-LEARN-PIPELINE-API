End-to-End Telco Customer Churn Prediction Pipeline
This repository contains a complete, production-quality, end-to-end Machine Learning pipeline that predicts customer churn using the IBM Telco Customer Churn Dataset. The project is built using Python, Scikit-Learn's Pipeline and ColumnTransformer APIs, and exports a tuned model using joblib.

📁 Project Structure
.
├── data/
│   ├── Telco-Customer-Churn.csv  # Raw dataset (downloaded programmatically)
│   ├── test_features.csv         # Saved test split features
│   └── test_target.csv           # Saved test split labels
├── models/
│   └── best_model.joblib         # Exported best performing model pipeline
├── notebooks/
│   └── telco_churn_pipeline.ipynb# Executed Jupyter Notebook containing EDA and tuning
├── reports/
│   ├── confusion_matrix.png      # Saved evaluation confusion matrix plot
│   └── roc_curve.png             # Saved evaluation ROC Curve plot
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py         # Downloads and loads dataset from raw source
│   ├── pipeline.py               # Preprocessing pipelines (imputation, scaling, one-hot encoding)
│   ├── train.py                  # Model training, GridSearchCV, comparison, and export
│   └── evaluate.py               # Decoupled model evaluation, metrics logging, and plotting
├── requirements.txt              # Project package dependencies
└── README.md                     # Project documentation
⚙️ Requirements & Installation
Clone the repository (or download files).
Create a virtual environment:
python -m venv .venv
Activate the virtual environment:
Windows (CMD/PowerShell):
.venv\Scripts\activate
Linux/macOS:
source .venv/bin/activate
Install the dependencies:
pip install -r requirements.txt
🚀 Usage
The project is structured modularly. You can run individual stages via scripts or run the entire process in the Jupyter notebook.

1. Ingest Data
Downloads the Kaggle Telco Customer Churn CSV dataset to data/:

python src/data_ingestion.py
2. Train and Tune Models
Runs GridSearchCV on Logistic Regression and Random Forest using 5-fold cross-validation (optimizing F1 score), selects the best estimator, saves the test split, and exports the final pipeline to models/best_model.joblib:

python src/train.py
3. Evaluate the Model
Loads the exported model pipeline and evaluates it against the saved test set, logging metrics and saving the Confusion Matrix and ROC curve plots in the reports/ folder:

python src/evaluate.py
📊 Exploratory Data Analysis & Key Findings
Churn Target: The dataset is imbalanced with a Churn rate of 26.54% (Yes) and 73.46% (No).
Tenure: Churned customers have a significantly lower median tenure (~10 months) than retained customers (~38 months). This indicates that early-stage subscribers are highly susceptible to churn.
Monthly Charges: High monthly charges strongly correlate with churn. The median monthly charge for churned customers is around $80 compared to $65 for retained ones.
Contract Type: Customers on Month-to-month contracts churn at a substantially higher rate than those on one-year or two-year contracts.
Internet Service: Subscribers utilizing Fiber optic connections experience a much higher churn rate compared to DSL or those without internet service.
🔬 Model Comparison & GridSearchCV Results
We optimized F1-score during 5-fold cross-validation on the training split to select the best model (as churn is imbalanced and minimizing false negatives is critical for business retention).

Classifier	Best CV F1 Score	Best Hyperparameters
Logistic Regression	0.5994	C=10.0, penalty='l2', solver='lbfgs'
Random Forest	0.5810	max_depth=10, min_samples_split=2, n_estimators=50
Note: Logistic Regression achieved a higher CV F1-score and was exported as our final production model.

📈 Evaluation Metrics (Test Set)
On the unseen test split (20% of data, 1,409 samples), the final exported Logistic Regression pipeline achieved the following results:

Accuracy: 0.8055
Precision: 0.6572
Recall: 0.5588
F1 Score: 0.6040
ROC-AUC: 0.8411
Confusion Matrix
The confusion matrix is saved to reports/confusion_matrix.png:

True Negatives (Retained correctly predicted): 923
False Positives (Retained predicted as Churn): 112
False Negatives (Churned predicted as Retained): 162
True Positives (Churned correctly predicted): 212
ROC Curve
The Receiver Operating Characteristic (ROC) curve is saved to reports/roc_curve.png, showcasing excellent discriminative power with an Area Under the Curve (AUC) of 0.8411.
