import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_loader(file_path):
    """Loads the dataset from a file."""
    return pd.read_csv(file_path)

def overview_data(df):
    """Returns an overview of the dataset, including structure and data types."""
    print("\nDataset Overview:")
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())

def summary_statistics(df):
    """Displays summary statistics for numerical columns."""
    print("\nSummary Statistics:")
    print(df.describe())

def visualize_numerical_distribution(df, numerical_columns):
    """Visualizes the distribution of numerical columns."""
    for col in numerical_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.show()

def visualize_categorical_distribution(df, categorical_columns):
    """Visualizes the distribution of categorical columns."""
    for col in categorical_columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.show()

def correlation_analysis(df, numerical_columns):
    """Plots a heatmap of correlations between numerical columns."""
    plt.figure(figsize=(10, 6))
    correlation_matrix = df[numerical_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix")
    plt.show()

def identify_missing_values(df):
    """Identifies missing values in the dataset."""
    missing = df.isnull().sum()
    print("\nMissing Values:")
    print(missing[missing > 0])

def detect_outliers(df, numerical_columns):
    """Uses box plots to detect outliers in numerical columns."""
    for col in numerical_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, y=col)
        plt.title(f"Box Plot for {col}")
        plt.show()