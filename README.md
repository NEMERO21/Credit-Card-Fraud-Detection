
# Credit Card Fraud Detection Dashboard

This project is a Streamlit-based web application for Credit Card Fraud Detection. The dashboard allows users to explore the dataset, perform various exploratory data analyses (EDA), train a Logistic Regression model, and evaluate its performance.

## Features

- **Data Loading**: Load the Credit Card Fraud Detection dataset.
- **Exploratory Data Analysis (EDA)**: 
  - Show raw data
  - Display summary statistics
  - Show class distribution
  - Display correlation heatmap
- **Model Training and Evaluation**: 
  - Train a Logistic Regression model
  - Evaluate the model's performance
  - Display evaluation metrics and confusion matrix

## Files

- **main.ipynb**: Contains the raw code for credit card fraud detection and model training.
- **app.py**: Streamlit app for the interactive dashboard.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection-dashboard.git
   cd credit-card-fraud-detection-dashboard
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload the dataset:**
   Place the dataset file (`creditcard.csv`) in the `dataset` directory.

2. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Explore the data and train the model:**
   - Use the sidebar options to show raw data, summary statistics, class distribution, and correlation heatmap.
   - Train the Logistic Regression model and evaluate its performance.

## Dataset

The dataset used in this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

## Credits

- **Developed by:** Shrestha Pundir
- **Dataset Source:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
