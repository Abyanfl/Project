
# Loan Approval Prediction Using Machine Learning

## Project Overview

This project aims to develop a machine learning model to automate loan approval decisions for fintech banks, such as "GrowEasy," by predicting whether a loan applicant will be approved or rejected. The model enhances efficiency, reduces risk, and ensures accurate predictions by balancing the approval of eligible applicants and the rejection of high-risk ones.

The primary goal is to:
- Automate and accelerate loan approval processes.
- Achieve high accuracy and recall, minimizing missed opportunities and financial losses due to defaults.
- Provide a user-friendly interface for real-time predictions via a deployed web app.

The dataset used is the "Loan Approval Prediction Dataset" from [Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data), containing 4,269 records with 11 features and a target variable (`loan_status`).

## Repository Structure

```
├── Loan_approval.ipynb      # Main Jupyter notebook for data analysis, preprocessing, modeling, and evaluation
├── inference.ipynb        # Inference notebook for testing the model on new data
├── xgb_pipeline_model.pkl      # Saved XGBoost model pipeline
├── loan_approval_dataset.csv   # Dataset used for the project
├── README.md                   # Project documentation
```

## Dataset Description

The dataset comprises 4,269 rows and 12 columns (11 features + 1 target after dropping `loan_id`):
- **Numerical Features (9)**: `no_of_dependents`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score`, `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`.
- **Categorical Features (2)**: `education` (Graduate, Not Graduate), `self_employed` (Yes, No).
- **Target Variable**: `loan_status` (Approved, Rejected).
- **Characteristics**: No missing values or duplicates. The dataset is slightly imbalanced (2,656 Approved vs. 1,613 Rejected).
- **Additional Features**: Created `loan_income_ratio` and `total_assets` during EDA for enhanced analysis.

## Methodology

### 1. Data Preprocessing
- **Feature Engineering**: Added `loan_income_ratio` and `total_assets` to capture financial relationships.
- **Outlier Handling**: Applied capping to manage extreme values in numerical features.
- **Encoding**: Used OneHotEncoder for categorical features and StandardScaler for numerical features.
- **Class Imbalance**: Addressed using SMOTENC to oversample the minority class (Rejected).

### 2. Model Development
Five models were evaluated:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- XGBoost

**Pipeline**: A scikit-learn pipeline integrated preprocessing (ColumnTransformer) and modeling. Hyperparameter tuning was performed using RandomizedSearchCV, with XGBoost selected as the best model.

### 3. Model Evaluation
- **Metrics**: Focused on F1-score (primary), recall, precision, accuracy, and ROC AUC, with cross-validation for robustness.
- **Results** (XGBoost, best model):
  - Test F1-score: 0.9803
  - Test Recall: 0.9757
  - Test Precision: 0.9849
  - ROC AUC: 0.9976
  - Cross-validation Recall: ~0.9877
- **Threshold Adjustment**: Set decision threshold to 0.3 to optimize recall while maintaining a high F1-score.

### 4. Model Deployment
- The XGBoost model was saved as `xgb_pipeline_model.pkl`.
- Deployed as a Streamlit web app on Hugging Face Spaces: [Model Deployment](https://huggingface.co/spaces/Abyanfl/Model-Prediction).
- An inference notebook (`P1M2_ABYAN_inf.ipynb`) demonstrates predictions on new data.

## Key Findings

### Model Performance
- **XGBoost** outperformed other models with an F1-score of 0.9803, balancing high recall (0.9757) and precision (0.9849).
- The model reduced overfitting compared to the initial version by simplifying decision trees, increasing regularization, and reducing the number of trees.
- A decision threshold of 0.3 improved recall, ensuring more eligible applicants are approved without sacrificing precision.

### Business Insights
- **CIBIL Score**: Applicants with scores above 700 are significantly more likely to be approved, indicating strong creditworthiness.
- **Loan-to-Income Ratio**: A critical feature for assessing repayment capacity; high ratios correlate with rejections.
- **Education and Self-Employment**: These factors have minimal impact on loan approval decisions.

## Recommendations

### For Fintech Banks (e.g., GrowEasy)
- **Adopt the Model**: Use the Streamlit app for fast, automated loan approvals, reducing manual review time.
- **Risk Management**: Leverage high precision to minimize risky approvals, protecting against defaults.
- **Policy Adjustments**: Prioritize applicants with CIBIL scores >700 and monitor loan-to-income ratios to set lending thresholds.
- **Scalability**: Use the web app to handle increasing digital loan applications in Indonesia.

### For Model Improvement
- **Additional Features**: Incorporate data like marital status or debt history to enhance risk detection.
- **Threshold Optimization**: Experiment with thresholds between 0.2–0.4 to further improve F1-score.
- **Ensemble Methods**: Combine XGBoost with Random Forest for potential accuracy gains.
- **Enhanced App**: Display feature importance (e.g., CIBIL score impact) in the web app for transparency.
- **Continuous Learning**: Update the model with real-world loan outcomes to maintain accuracy.

## Installation and Usage

### Requirements
- Python 3.12.9
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, imblearn, xgboost, joblib, streamlit

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imblearn xgboost joblib streamlit
```



### Inference
1. Open `inference.ipynb`.
2. Load the saved model (`xgb_pipeline_model.pkl`).
3. Use the notebook to make predictions on new applicant data.

### Deployment
The model is deployed on Hugging Face Spaces: [Model Deployment](https://huggingface.co/spaces/Abyanfl/Model-Prediction). Access the Streamlit app to input applicant data and receive instant approval predictions.

## Limitations
- **Class Imbalance**: Despite using SMOTENC, the slight imbalance (62% Approved vs. 38% Rejected) may affect model generalization.
- **Feature Scope**: Limited to 11 features; additional data (e.g., debt history) could improve predictions.
- **Threshold Sensitivity**: The 0.3 threshold optimizes recall but may require further tuning for specific business needs.

## Future Work
- Collect additional features (e.g., debt-to-income ratio, employment history) to enhance model performance.
- Explore ensemble methods or neural networks for potential accuracy improvements.
- Enhance the Streamlit app with visualizations of feature importance for better decision-making.
- Scale the app to handle high volumes of applications as digital lending grows.
- Implement continuous model retraining with real-world loan performance data.

## Author
- **Name**: Abyan Naufal
- **Contact**: [Your contact information or GitHub profile]

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
