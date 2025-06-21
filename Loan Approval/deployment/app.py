# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from eda import perform_eda
from prediction import load_model, predict_attrition
import plotly.graph_objects as go

# Load the model at startup
model = load_model('xgb_pipeline_model.pkl')

# Load dataset for EDA
dataset_path = 'loan_approval_dataset.csv'

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Inference"])

# EDA Page
if page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    st.header("Loan Approval Dataset Insights")
    
   # Perform EDA
    try:
        eda_insights = perform_eda(dataset_path)
        df = pd.read_csv(dataset_path)
        
        # Dataset Overview (keep as text)
        st.text("Dataset Overview:")
        st.text(eda_insights['dataset_overview'])
        
        # Loan Status Distribution (already plotted in eda.py, keep original)
        st.text("Loan Status Distribution:")
        st.text(eda_insights['loan_status_dist'])
        st.image(f"data:image/png;base64,{eda_insights['loan_status_plot']}", caption="Loan Status Distribution")
        
        # Average Annual Income by Loan Status (new Plotly chart)
        st.header("Average Annual Income by Loan Status")
        avg_income = df.groupby('loan_status')['income_annum'].mean().reset_index()
        avg_income['loan_status'] = avg_income['loan_status'].map({0: 'Rejected', 1: 'Approved'})
        fig1 = px.bar(avg_income, x='loan_status', y='income_annum', 
                      title="Average Annual Income by Loan Status",
                      labels={'loan_status': 'Loan Status', 'income_annum': 'Average Annual Income (INR)'},
                      color='loan_status', color_discrete_map={'Rejected': '#FF6384', 'Approved': '#36A2EB'})
        st.plotly_chart(fig1)
        
        # Numerical Columns Skewness (new Plotly chart)
        st.header("Numerical Columns Skewness")
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        skewness = df[numerical_cols].skew().reset_index()
        skewness.columns = ['Column', 'Skewness']
        fig2 = px.bar(skewness, x='Column', y='Skewness', 
                      title="Skewness of Numerical Columns",
                      labels={'Column': 'Numerical Columns', 'Skewness': 'Skewness'},
                      color='Skewness', color_continuous_scale='Viridis')
        st.plotly_chart(fig2)
        
        # Education vs Loan Status (new Plotly chart)
        st.header("Education vs Loan Status")
        education_loan = pd.crosstab(df['education'], df['loan_status']).reset_index()
        education_loan.columns = ['education', 'Rejected', 'Approved']
        fig3 = go.Figure(data=[
            go.Bar(name='Rejected', x=education_loan['education'], y=education_loan['Rejected'], marker_color='#FF6384'),
            go.Bar(name='Approved', x=education_loan['education'], y=education_loan['Approved'], marker_color='#36A2EB')
        ])
        fig3.update_layout(barmode='stack', title="Education vs Loan Status",
                           xaxis_title="Education", yaxis_title="Count")
        st.plotly_chart(fig3)
        
        # CIBIL Score Distribution (keep original)
        st.header("CIBIL Score Distribution by Loan Status")
        st.image(f"data:image/png;base64,{eda_insights['cibil_score_plot']}", caption="CIBIL Score Distribution by Loan Status")
        
    except Exception as e:
        st.error(f"Error loading EDA insights: {e}")

elif page == "Inference":
    st.title("Loan Approval Prediction")
    
    # Create a form for user input
    st.header("Input Applicant Data")
    with st.form(key='loan_form'):
        # Collect user input for all features
        no_of_dependents = st.slider("Number of Dependents", min_value=0, max_value=5, value=2, step=1)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        income_annum = st.number_input("Annual Income", min_value=200000, max_value=9900000, value=5000000)
        loan_amount = st.number_input("Loan Amount", min_value=100000, max_value=40000000, value=5000000)  # Added
        loan_term = st.slider("Loan Term (Years)", min_value=2, max_value=20, value=10, step=2)
        cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=750, step=1)
        residential_assets_value = st.number_input("Residential Assets Value", min_value=-100000, max_value=29100000, value=7000000)
        commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, max_value=19400000, value=3000000)
        luxury_assets_value = st.number_input("Luxury Assets Value", min_value=300000, max_value=39200000, value=12000000)
        bank_asset_value = st.number_input("Bank Asset Value", min_value=0, max_value=14700000, value=4000000)

        # Submit button
        submit_button = st.form_submit_button(label="Predict Loan Approval")

    # Process the form submission
    if submit_button:
        # Create a dictionary with the input data
        form_data = {
            'no_of_dependents': no_of_dependents,
            'education': education,
            'self_employed': self_employed,
            'income_annum': income_annum,
            'loan_amount': loan_amount,  # Added
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets_value,
            'commercial_assets_value': commercial_assets_value,
            'luxury_assets_value': luxury_assets_value,
            'bank_asset_value': bank_asset_value
        }

        # Convert to DataFrame
        new_data = pd.DataFrame([form_data])

        # Make prediction
        try:
            predictions = predict_attrition(model, new_data)
            prediction = predictions[0]  # Take the first (and only) prediction
            probability = model.predict_proba(new_data)[0][1] if prediction == "Approved" else model.predict_proba(new_data)[0][0]

            # Display the prediction
            st.header("Prediction Result")
            st.write(f"**Predicted Loan Status:** {prediction}")
            st.write(f"**Probability:** {probability:.2%}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")