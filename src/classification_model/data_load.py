import os
import pandas as pd

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "raw",
    "Telco_Customer_Churn.csv"
)

def load_data() -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset into a DataFrame.
    Returns:
        pd.DataFrame: Raw data including customer features and churn label.
    """
    # Ensure the file exists at the specified path
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}.")
    df = pd.read_csv(DATA_PATH)
    return df