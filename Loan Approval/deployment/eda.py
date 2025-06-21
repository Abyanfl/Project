import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np

def perform_eda(dataset_path):
    df = pd.read_csv(dataset_path)
    df.drop(columns=['loan_id'], errors='ignore', inplace=True)
    
    eda_insights = {}
    
    # Dataset Overview
    eda_insights['dataset_overview'] = f"Dataset Shape: {df.shape}\n" \
                                      f"Columns: {list(df.columns)}\n" \
                                      f"Missing Values:\n{df.isnull().sum().to_string()}"
    
    # Loan Status Distribution
    loan_status_counts = df['loan_status'].value_counts().to_string()
    eda_insights['loan_status_dist'] = f"Loan Status Distribution:\n{loan_status_counts}"
    plt.figure(figsize=(6, 4))
    sns.countplot(x='loan_status', data=df)
    plt.title('Loan Status Distribution')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    eda_insights['loan_status_plot'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # Average Annual Income by Loan Status
    avg_income = df.groupby('loan_status')['income_annum'].mean().to_string()
    eda_insights['avg_income'] = f"Average Annual Income by Loan Status:\n{avg_income}"
    
    # Numerical Columns Skewness
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    skewness = df[numerical_cols].skew().to_string()
    eda_insights['skewness'] = f"Skewness of Numerical Columns:\n{skewness}"
    
    # Education vs Loan Status
    education_loan = pd.crosstab(df['education'], df['loan_status']).to_string()
    eda_insights['education_loan'] = f"Education vs Loan Status:\n{education_loan}"
    
    # CIBIL Score Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='cibil_score', hue='loan_status', multiple='stack')
    plt.title('CIBIL Score Distribution by Loan Status')
    plt.xlabel('CIBIL Score')
    plt.ylabel('Count')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    eda_insights['cibil_score_plot'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
               
    # No. of Dependents vs Loan Status
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='no_of_dependents', hue='loan_status')
    plt.title('No. of Dependents vs Loan Status')
    plt.xlabel('Number of Dependents')
    plt.ylabel('Count')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    eda_insights['dependents_loan_plot'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # Loan Term vs Loan Status
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='loan_term', hue='loan_status', multiple='stack', bins=10)
    plt.title('Loan Term Distribution by Loan Status')
    plt.xlabel('Loan Term (Years)')
    plt.ylabel('Count')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    eda_insights['loan_term_plot'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return eda_insights