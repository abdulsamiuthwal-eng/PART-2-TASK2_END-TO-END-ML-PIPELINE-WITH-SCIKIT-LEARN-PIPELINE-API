import os
import requests
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_PATH = os.path.join(DATA_DIR, "Telco-Customer-Churn.csv")

def download_data(url: str = DATA_URL, dest_path: str = DATA_PATH) -> str:
    """Downloads the Telco Customer Churn dataset from a URL to a local destination.

    Args:
        url: The URL to download the dataset from.
        dest_path: The local destination file path.

    Returns:
        The destination path where the dataset is stored.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    print(f"Downloading dataset from: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    with open(dest_path, "wb") as f:
        f.write(response.content)
        
    print(f"Dataset successfully saved to: {dest_path}")
    return dest_path

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Loads the downloaded dataset into a pandas DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at {path}. Run download_data() first.")
    
    df = pd.read_csv(path)
    print(f"Dataset loaded. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    download_data()
    df = load_data()
    print("Columns in dataset:")
    print(df.columns.tolist())
