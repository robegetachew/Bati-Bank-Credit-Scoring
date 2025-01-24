# Credit Scoring Model for Bati Bank

## Overview
Bati Bank, a leading financial institution, is introducing a "Buy Now, Pay Later" service in collaboration with an eCommerce platform. This project aims to develop a robust **Credit Scoring Model** to assess potential borrowers' risk of default. The solution is designed to ensure compliance with the Basel II Capital Accord and provide actionable insights for effective risk management.

## Features
- Ingestion and preprocessing of transactional data from the eCommerce platform.
- Exploratory Data Analysis (EDA) to identify key predictors of default risk.
- Development of a predictive model for risk scoring.
- Translation of risk probabilities into actionable credit scores.
- Insights to optimize loan amounts and durations.

## Tech Stack
- **Python**: Data preprocessing, EDA, and model development.
- **Pandas & NumPy**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning model development.
- **Telethon**: Data ingestion from Telegram channels (if applicable).

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/1Light/kaim-week-6.git
   cd kaim-week-6
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Data Preprocessing:**
   - Run the script to ingest and preprocess transactional data:
     ```bash
     python preprocess_data.py
     ```
2. **EDA:**
   - Visualize key trends and patterns:
     ```bash
     python eda.py
     ```
3. **Model Training:**
   - Train the credit scoring model:
     ```bash
     python train_model.py
     ```

## Project Structure
```plaintext
├── data/                     # Raw and processed datasets
├── notebooks/                # Jupyter notebooks for EDA and modeling
├── scripts/                  # Python scripts for preprocessing and model training
├── requirements.txt          # Dependencies
├── README.md                 # Project overview
```

## Next Steps
- Finalize the proxy variable for risk categorization.
- Train and evaluate credit scoring models.
- Implement risk probability frameworks.
- Develop optimization models for loan amounts and durations.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any suggestions or improvements.

## License
This project is licensed under the [MIT License](LICENSE)."# Bati-Bank-Credit-Scoring" 
