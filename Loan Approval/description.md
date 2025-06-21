# Loan Approval Prediction Project

## Repository Outline
This repository contains all files related to the Phase 1 Milestone 2 project for the Hacktiv8 Data Science Fulltime Program (Batch HCK-027). The contents are:

1. description.md: This file, providing an overview of the project, problem background, data, methods, and stacks.
2. loan approval.ipynb: Jupyter notebook containing data loading, exploratory data analysis (EDA), feature engineering, model training, evaluation, and saving for the loan approval prediction model.
3. inference.ipynb: Jupyter notebook for model inference, testing the trained model on new, raw data.
4. xgb_pipeline_model.pkl: Saved XGBoost model pipeline, including preprocessing and classifier.
5. deployment: Folder containing deployment files (e.g., `app.py`) for the web app 
6. url.txt: File containing the dataset URL and deployment URL (https://huggingface.co/spaces/Abyanfl/Model-Prediction).

## Problem Background
Loan approval decisions are critical for financial institutions to balance profitability and risk. Incorrectly approving high-risk applicants can lead to financial losses due to defaults, while rejecting eligible applicants results in missed business opportunities. In Indonesia, where access to credit is vital for small businesses and individuals, manual loan evaluations are time-consuming and prone to errors. This project addresses the need for an automated, data-driven system to predict whether a loan applicant is likely to be approved or rejected based on their financial and personal attributes, improving efficiency and accuracy for banks like "GrowEasy."

## Project Output
The project delivers a machine learning model (XGBoost) that predicts loan approval status (`Approved` or `Rejected`) for new applicants. The model is saved as `xgb_pipeline_model.pkl` and can be used via a Jupyter notebook (`P1M2_ABYAN_inf.ipynb`) for inference. A web app deployment is planned to allow real-time predictions, enabling banks to input applicant data and receive instant approval decisions.

## Data
The dataset used is the "Loan Approval Dataset" (sourced from [Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data). It contains 4,269 rows and 12 columns (11 features + 1 target after dropping `loan_id`):
- Features:
  - Numerical (9): `no_of_dependents`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score`, `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`.
  - Categorical (2): `education` (Graduate, Not Graduate), `self_employed` (Yes, No).
- Target: `loan_status` (Approved, Rejected).
- Characteristics: No missing values or duplicates. The dataset is slightly imbalanced (2,656 Approved vs. 1,613 Rejected), addressed using SMOTENC during modeling.
- Additional Features: Created `loan_income_ratio` and `total_assets` during EDA for deeper analysis.

## Method
The project employs supervised learning for binary classification. The methodology includes:
1. Data Preprocessing: Outlier capping, standard scaling for numerical features, one-hot encoding for categorical features, and SMOTENC for handling class imbalance.
2. Modeling: Five algorithms were tested (KNN, SVM, Decision Tree, Random Forest, XGBoost). XGBoost was selected as the best model due to its high F1-score (0.9841) and recall (0.9813) on the test set.
3. Evaluation: Models were evaluated using F1-score, recall, precision, accuracy, and ROC AUC, with cross-validation to ensure robustness. Hyperparameter tuning was performed using RandomizedSearchCV to optimize XGBoost.
4. Inference: The trained model is tested on new, raw data to predict loan approval status.
5. Deployment: A web app (in progress) will allow users to input applicant data and receive predictions.

## Stacks
- Programming Language: Python 3.11.7
- Libraries:
  - Data Processing: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn` (preprocessing, models, metrics, pipelines), `xgboost`, `imblearn` (SMOTENC)
  - Model Saving: `joblib`
- Tools: Jupyter Notebook, Git, GitHub Classroom, HuggingFace
- Environment: Anaconda (local development)

## Reference
- Dataset: [Loan Approval Dataset on Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data).
- Deployment: [HuggingFace](https://huggingface.co/spaces/Abyanfl/Model-Prediction).
- Conceptual References:
  - [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
  - [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
  - [Imbalanced-Learn Documentation](https://imbalanced-learn.org/stable/)
- Markdown Guide: [GitHub Markdown Syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

---

Installation Instructions (for running the project locally):
1. Clone the repository: `git clone`
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib`
3. Run the notebooks: Open `P1M2_ABYAN.ipynb` for training and `P1M2_ABYAN_inf.ipynb` for inference in Jupyter Notebook.
4. Ensure `xgb_pipeline_model.pkl` is in the same directory for inference.
5. For deployment, navigate to the `deployment` folder and follow instructions in `app.py`.

**Contact**: For questions, contact Abyan Naufal at [abyanaufal@gmail.com].