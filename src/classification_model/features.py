import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame):
    """
    Clean and transform the raw Telco churn data into features and target for modeling.
    Returns:
        X (pd.DataFrame): Preprocessed feature matrix
        y (pd.Series): Target array (Churn: 1 or 0)
    """
    # Make a copy to avoid modifying original DataFrame
    data = df.copy()

    # 1. Drop customerID if present (not useful for prediction)
    if 'customerID' in data.columns:
        data = data.drop('customerID', axis=1)

    # 2. Handle missing or blank values in TotalCharges by converting to numeric and filling NAs
    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        # Fill missing TotalCharges with 0 (assume no charges for new customers)
        data['TotalCharges'] = data['TotalCharges'].fillna(0)

    # 3. Encode the target variable 'Churn' as 0/1
    # If Churn exists in DataFrame, separate it as y
    if 'Churn' in data.columns:
        y = data['Churn'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1'] else 0)
        data = data.drop('Churn', axis=1)
    else:
        raise KeyError("No 'Churn' column found in data for target variable.")

    # 4. One-hot encode categorical features
    # Identify categorical columns (object or category dtype after removing target)
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    X = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    # (drop_first=True avoids one dummy column per category to prevent collinearity)

    # 5. Scale numerical features
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    # Fit scaler on the numeric columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y