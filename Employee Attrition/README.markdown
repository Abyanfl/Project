
# Employee Attrition Using Machine Learning

## Project Overview

This project focuses on predicting employee attrition within a company using machine learning approach. The main goal is to develop a model that can:

- Predict whether an employee is likely to leave the organization (Attrition = Yes).
- Prioritize high recall for the minority class (employees who leave) while maintaining a balanced performance using F1-score.
- Provide actionable insights for HR teams to identify key factors contributing to employee turnover and improve retention strategies.

The dataset used is the IBM HR Analytics Employee Attrition & Performance dataset (`WA_Fn-UseC_-HR-Employee-Attrition.csv`), which contains 1,470 employee records with 35 features, including demographic, job-related, and satisfaction metrics.

## Project Objectives

- **Model Development**: Build and evaluate multiple machine learning models to predict attrition, emphasizing recall for the minority class.
- **Feature Importance**: Identify the most influential factors affecting employee attrition to provide actionable business insights.
- **Model Deployment**: Deploy the best-performing model for practical use in identifying at-risk employees.

## Repository Structure

```
├── Attrition.ipynb           # Main Jupyter notebook for data analysis, preprocessing, modeling, and evaluation
├── P1M2_abyan_inference.ipynb # Inference notebook for testing the model on unseen data
├── best_model_rf.pkl         # Saved Random Forest model
├── preprocessor.pkl          # Saved preprocessing pipeline
├── WA_Fn-UseC_-HR-Employee-Attrition.csv # Dataset used for the project
├── README.md                 # Project documentation
```

## Dataset Description

The dataset consists of 1,470 rows and 35 columns with no missing values. Key features include:

- **Numerical Features**: Age, DailyRate, DistanceFromHome, MonthlyIncome, TotalWorkingYears, etc.
- **Categorical Features**: BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime, etc.
- **Target Variable**: Attrition (Yes/No), with an imbalanced distribution (1,233 No vs. 237 Yes).

## Methodology

### 1. Data Preprocessing
- **Feature Selection**: Dropped irrelevant features (e.g., EmployeeCount, StandardHours, Over18, EmployeeNumber) due to no predictive value.
- **Encoding**: Applied OneHotEncoder for categorical features and StandardScaler for numerical features.
- **Handling Imbalance**: Used SMOTENC to oversample the minority class (Attrition = Yes).

### 2. Model Development
Five models were evaluated:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- XGBoost

**Pipeline**: A scikit-learn pipeline was used, integrating preprocessing (ColumnTransformer) and model training. Hyperparameter tuning was performed using GridSearchCV.

### 3. Model Evaluation
- **Metrics**: Focused on recall for Attrition = Yes, F1-score, and cross-validation accuracy.
- **Results**:
  - **Random Forest** (Best Model):
    - Cross-validation Accuracy: 85.99%
    - Test Accuracy: 84.35%
    - Recall (Attrition = Yes): 40.43%
    - F1-score: 45.24%
  - **XGBoost**: Higher test accuracy (85.03%) but lower recall (34.04%).
  - **KNN**: Highest recall (51.06%) but poor precision (26.67%).

### 4. Model Deployment
- The Random Forest model was saved as `best_model_rf.pkl` and deployed on Hugging Face Spaces: [Model Deployment](https://huggingface.co/spaces/Abyanfl/ModelDeployment).
- An inference notebook (`P1M2_abyan_inference.ipynb`) demonstrates predictions on unseen data.

## Key Findings

### Model Performance
- Random Forest provided the best trade-off between recall and precision, making it suitable for deployment.
- The recall of 40.43% for Attrition = Yes indicates room for improvement in detecting employees likely to leave.

### Business Insights
- **DistanceFromHome**: Employees living farther from work are more likely to leave, possibly due to commute stress.
- **BusinessTravel & OverTime**: Frequent travel and overtime work are strongly associated with attrition.
- **MonthlyIncome & YearsSinceLastPromotion**: Skewed distributions suggest potential issues with salary competitiveness and promotion timing.

## Recommendations

### For HR
- **Retention Strategies**:
  - Offer flexible work arrangements (e.g., remote work) for employees with long commutes.
  - Reduce overtime and travel demands where possible.
  - Review promotion cycles and ensure competitive salaries.
- **Proactive Interventions**: Use the model to flag at-risk employees for targeted engagement programs.

### For Model Improvement
- **Feature Transformation**: Apply log transformation to skewed features (e.g., MonthlyIncome, YearsSinceLastPromotion).
- **Hyperparameter Tuning**: Experiment with `class_weight='balanced'` in Random Forest or adjust decision thresholds.
- **Alternative Techniques**: Explore ADASYN, SMOTE variants, or cost-sensitive learning to boost recall.
- **Feature Selection**: Use Random Forest feature importance to reduce dimensionality.

## Installation and Usage

### Requirements
- Python 3.12.9
- Libraries: pandas, numpy, scikit-learn, imblearn, xgboost, matplotlib, seaborn, plotly, joblib

Install dependencies:
```bash
pip install -r requirements.txt
```

## Deployment
The model is deployed on Hugging Face Spaces: [Model Deployment](https://huggingface.co/spaces/Abyanfl/ModelDeployment). Follow the link to interact with the model.

## Limitations
- **Recall Limitation**: The recall for Attrition = Yes (40.43%) is moderate, indicating some at-risk employees may be missed.
- **Data Imbalance**: The highly imbalanced dataset (16% Attrition = Yes) challenges model performance.
- **Feature Skewness**: Skewed features may limit the performance of certain models (e.g., KNN, SVM).

## Future Work
- Integrate additional data sources (e.g., employee feedback surveys) to enhance predictive power.
- Experiment with ensemble methods or deep learning for improved performance.
- Develop a user-friendly interface for HR teams to interact with the model.

## Author
- **Name**: Abyan Naufal
- **Contact**: [Your contact information or GitHub profile
