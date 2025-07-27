# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
import kagglehub
import joblib
import os 
# Page config
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")

# Load dataset
path = kagglehub.dataset_download("nikhilbhosle/employee-attrition-uncleaned-dataset")
df = pd.read_csv(path + '/Emp_attrition_csv.csv')
df.drop(['Employee ID'], axis=1, inplace=True)
df.drop_duplicates(inplace=True)

# Handle missing values
cat_cols = df.select_dtypes(include='object').columns
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_imputer = SimpleImputer(strategy='mean')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Encode categorical variables
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare data
x = df.drop('Attrition', axis=1)
y = df['Attrition']

# Undersample to balance
rus = RandomUnderSampler(random_state=42)
X_balanced, y_balanced = rus.fit_resample(x, y)

# Scale features
scaler = StandardScaler()
X_balanced_scaled = scaler.fit_transform(X_balanced)

# Train model
#rf_model = RandomForestClassifier(n_estimators=500, max_depth=20)
#rf_model.fit(X_balanced_scaled, y_balanced)
base_dir = r"C:\Users\HP\Desktop\New folder"

# Load saved objects
model = joblib.load(os.path.join(base_dir, "random_forest.pkl"))
# UI Title
st.title("🔍 Employee Attrition Prediction App")

# Input form
st.subheader("Enter employee details:")

# Generate input fields dynamically
user_input = {}
for col in x.columns:
    if col in cat_cols:
        le = label_encoders[col]
        options = list(le.classes_)
        selected = st.selectbox(f"{col}:", options)
        user_input[col] = le.transform([selected])[0]
    else:
        val = st.number_input(f"{col}:", int(df[col].min()), int(df[col].max()), int(df[col].mean()))
        user_input[col] = val

# Predict
if st.button("Predict Attrition"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ The employee is likely to leave.")
    else:
        st.success("✅ The employee is likely to stay.")

