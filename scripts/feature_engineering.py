import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Aggregate Features
def create_aggregate_features(data, customer_id_col, amount_col):
    """Create aggregate features for each customer."""
    aggregate_features = data.groupby(customer_id_col).agg(
        total_transaction_amount=(amount_col, 'sum'),
        average_transaction_amount=(amount_col, 'mean'),
        transaction_count=(amount_col, 'count'),
        std_transaction_amount=(amount_col, 'std')
    ).reset_index()
    return aggregate_features

# Extract Features from Timestamp
def extract_time_features(data, timestamp_col):
    """Extract time-based features from a timestamp column."""
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data['transaction_hour'] = data[timestamp_col].dt.hour
    data['transaction_day'] = data[timestamp_col].dt.day
    data['transaction_month'] = data[timestamp_col].dt.month
    data['transaction_year'] = data[timestamp_col].dt.year
    return data

# Encode Categorical Variables
def encode_categorical(data, cat_cols, method='onehot'):
    """
    Encode categorical variables.
    :param data: DataFrame with categorical columns
    :param cat_cols: List of categorical columns to encode
    :param method: 'onehot' for One-Hot Encoding or 'label' for Label Encoding
    """
    if method == 'onehot':
        encoder = OneHotEncoder(sparse_output=False, drop='first')  # Updated to sparse_output
        encoded_data = pd.DataFrame(encoder.fit_transform(data[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))
        data = pd.concat([data.reset_index(drop=True), encoded_data], axis=1).drop(columns=cat_cols)
    elif method == 'label':
        encoder = LabelEncoder()
        for col in cat_cols:
            data[col] = encoder.fit_transform(data[col])
    return data

# Handle Missing Values
def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values.
    :param strategy: 'mean', 'median', 'most_frequent', or 'remove'
    :param columns: List of columns to apply the strategy (default is all columns)
    """
    if strategy == 'remove':
        data = data.dropna(subset=columns)
    else:
        imputer = SimpleImputer(strategy=strategy)
        data[columns] = imputer.fit_transform(data[columns])
    return data

# Normalize or Standardize Numerical Features
def scale_numerical_features(data, num_cols, method='normalize'):
    """
    Scale numerical features.
    :param method: 'normalize' (Min-Max Scaling) or 'standardize' (Z-score)
    """
    scaler = MinMaxScaler() if method == 'normalize' else StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])
    return data

