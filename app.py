import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Set up the page layout
st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide")

# Load data function
@st.cache
def load_data(file_path): 
    df = pd.read_csv(file_path)
    return df

# Preprocess data function
@st.cache
def preprocess_data(df):
    fraud = df[df['Class'] == 1]
    normal = df[df['Class'] == 0]
    legit_sample = normal.sample(n=len(fraud))
    new_dataset = pd.concat([legit_sample, fraud], axis=0)
    X = new_dataset.drop(columns='Class')
    y = new_dataset['Class']
    return X, y

# Train model function
@st.cache(allow_output_mutation=True)
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Load the dataset
df = load_data('dataset/creditcard.csv')

# Sidebar options
st.sidebar.title("Options")
show_raw_data = st.sidebar.checkbox("Show Raw Data")
show_summary_stats = st.sidebar.checkbox("Show Summary Statistics")
show_class_distribution = st.sidebar.checkbox("Show Class Distribution")
show_correlation_heatmap = st.sidebar.checkbox("Show Correlation Heatmap")
show_model_training = st.sidebar.checkbox("Train and Evaluate Model")

# Main layout
st.title("Credit Card Fraud Detection Dashboard")

# Homepage section
st.header("Welcome to the Credit Card Fraud Detection Dashboard")
st.write("""
This dashboard allows you to explore the Credit Card Fraud Detection dataset, perform various exploratory data analyses (EDA), 
train a Logistic Regression model, and evaluate its performance. Use the options in the sidebar to interact with the data and visualize results.
""")
st.write("Dataset Source: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")

if show_raw_data:
    st.subheader("Raw Data")
    st.write(df.head())

if show_summary_stats:
    st.subheader("Summary Statistics")
    st.write(df.describe())

if show_class_distribution:
    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=df, ax=ax)
    st.pyplot(fig)

if show_correlation_heatmap:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

if show_model_training:
    st.subheader("Train and Evaluate Model")
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Display evaluation metrics
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("ROC AUC Score:", roc_auc_score(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Display confusion matrix
    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

# Footer
st.markdown("Developed with ❤️ by Shrestha Pundir")
