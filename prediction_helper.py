import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# 1. DYNAMIC PATH HANDLING (Crucial for Streamlit Cloud)
# This gets the directory where THIS file (prediction_helper.py) is located
current_dir = os.path.dirname(__file__)

# Combine the directory with the file name. 
# valid for your current setup where model_data.joblib is in the ROOT folder.
MODEL_PATH = os.path.join(current_dir, 'model_data.joblib')

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    cols_to_scale = model_data['cols_to_scale']
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find model file at {MODEL_PATH}. Ensure the file is uploaded to GitHub.")

def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                  delinquency_ratio, credit_utilization_ratio, num_open_accounts, 
                  residence_type, loan_purpose, loan_type):
    
    # Calculate derived features
    loan_to_income = loan_amount / income if income > 0 else 0
    
    # Create a dictionary with input values and dummy values for missing features
    # (These match the columns your model was trained on)
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_to_income,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        
        # Categorical encoding (matching your training data)
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        
        # Dummy fields (Required if your Scaler expects them)
        'number_of_dependants': 1, 
        'years_at_current_address': 1, 
        'zipcode': 1, 
        'sanction_amount': 1, 
        'processing_fee': 1, 
        'gst': 1, 
        'net_disbursement': 1, 
        'principal_outstanding': 1, 
        'bank_balance_at_application': 1, 
        'number_of_closed_accounts': 1, 
        'enquiry_count': 1 
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Scale only the columns that expect scaling
    # (We use .get() to avoid errors if a column is missing in input_data)
    if cols_to_scale:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Reorder columns to match the training data exactly
    df = df[features]

    return df

def calculate_credit_score(input_df, base_score=300, scale_length=600):
    # Use predict_proba instead of manual dot product
    # This works for XGBoost, Random Forest, AND Logistic Regression
    probability = model.predict_proba(input_df)[0][1]  # Probability of Default (Class 1)
    
    non_default_probability = 1 - probability
    
    # Scale the probability to a credit score (300-900)
    credit_score = base_score + (non_default_probability * scale_length)
    
    return probability, int(credit_score)

def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):
    
    input_df = prepare_input(
        age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
        delinquency_ratio, credit_utilization_ratio, num_open_accounts,
        residence_type, loan_purpose, loan_type
    )
    
    probability, credit_score = calculate_credit_score(input_df)
    
    # Rating Logic
    if 300 <= credit_score < 500:
        rating = 'Poor'
    elif 500 <= credit_score < 650:
        rating = 'Average'
    elif 650 <= credit_score < 750:
        rating = 'Good'
    elif 750 <= credit_score <= 900:
        rating = 'Excellent'
    else:
        rating = 'Undefined'

    return probability, credit_score, rating
