import streamlit as st
import pandas as pd
import joblib

# Load pipeline (includes preprocessing + model)
pipeline = joblib.load("loan_pipeline.pkl")

st.title("💳 Loan Defaulter Prediction App")
st.write("Enter customer details to check if they are likely to default.")

# Collect raw user inputs (not encoded!)
age = st.number_input('Age', min_value=18,max_value =80,value=40)
loannumber = st.number_input('Amount of loans' , min_value=1,value=4)
loanamount = st.number_input('Loan Amount' , min_value=10000,value=10000)
totaldue = st.number_input('Total due' , min_value=10000,value=15000)
termdays = st.number_input('Loan term(days)', min_value=15,max_value =90,value=30)
loan_approval_min = st.number_input('Loanapproval time(min)' , min_value=60.00,value=60.01)
early_payments = st.number_input('Early payment count' , min_value=0,value=2)
late_payments = st.number_input('Late payment count' , min_value=0,value=0)
payment_status_score=  st.slider('Payment status score' , min_value= -30.5,value=9.5)
repayment_behaviour_score =  st.slider('Repayment behaviour score' , min_value=-500.0,max_value =500.0,value=475.0)
bank_account_type = st.selectbox( "Account Type" , ['Current','Other', 'Savings'])
employment_status_clients = st.selectbox("Employment Type", ["Permanent", "Retired","Self-Employed","Student","Unemployed","unknown","Contract"])


# Put into dataframe (same as training BEFORE encoding)
features = pd.DataFrame([[age,loannumber, loanamount,totaldue, termdays,loan_approval_min,early_payments,late_payments,payment_status_score,repayment_behaviour_score,bank_account_type,employment_status_clients]],
columns=[ 'age','loannumber','loanamount','totaldue','termdays','loan_approval_min','early_payments','late_payments','payment_status_score','repayment_behaviour_score','bank_account_type','employment_status_clients'])

if st.button("Predict"):
    prediction = pipeline.predict(features)[0]  # pipeline does encoding internally
    
    if prediction == 0:   # 0 = defaulter
        st.error("🚨 This customer is likely to DEFAULT on the loan!")
    else:  # 1 = non-defaulter
        st.success("✅ This customer is likely to REPAY the loan.")



