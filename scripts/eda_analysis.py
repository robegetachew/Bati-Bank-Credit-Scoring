import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self, file_path):
        """Initialize the DataAnalyzer with the dataset."""
        self.file_path = file_path
        self.df = self._load_data()

    def _load_data(self):
        """Loads the dataset from a file."""
        return pd.read_csv(self.file_path)

    def overview(self):
        """Displays an overview of the dataset."""
        print("\nDataset Overview:")
        print(f"Number of Rows: {self.df.shape[0]}")
        print(f"Number of Columns: {self.df.shape[1]}")
        print("\nData Types:")
        print(self.df.dtypes)
        print("\nMissing Values:")
        print(self.df.isnull().sum())

    def summary_statistics(self):
        """Displays summary statistics for numerical columns."""
        print("\nSummary Statistics:")
        print(self.df.describe())

    def visualize_numerical_distribution(self, numerical_columns):
        """Visualizes the distribution of numerical columns."""
        for col in numerical_columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.show()

    def visualize_categorical_distribution(self, categorical_columns):
        """Visualizes the distribution of categorical columns."""
        for col in categorical_columns:
            plt.figure(figsize=(8, 4))
            sns.countplot(data=self.df, x=col, order=self.df[col].value_counts().index)
            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=45)
            plt.show()

    def correlation_analysis(self, numerical_columns):
        """Plots a heatmap of correlations between numerical columns."""
        plt.figure(figsize=(10, 6))
        correlation_matrix = self.df[numerical_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix")
        plt.show()

    def identify_missing_values(self):
        """Identifies missing values in the dataset."""
        missing = self.df.isnull().sum()
        print("\nMissing Values:")
        print(missing[missing > 0])

    def detect_outliers(self, numerical_columns):
        """Uses box plots to detect outliers in numerical columns."""
        for col in numerical_columns:
            plt.figure(figsize=(8, 4))
            sns.boxplot(data=self.df, y=col)
            plt.title(f"Box Plot for {col}")
            plt.show()

analyzer = DataAnalyzer("data.csv")
analyzer.overview()
analyzer.summary_statistics()
analyzer.visualize_numerical_distribution(["col1", "col2"])
analyzer.visualize_categorical_distribution(["col3", "col4"])
analyzer.correlation_analysis(["col1", "col2", "col5"])
analyzer.identify_missing_values()
analyzer.detect_outliers(["col1", "col2"])