import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv('student_mat.csv')

# BASIC DATA PREPROCESSING (adjust as needed)
df['pass'] = df['G3'] >= 10      # Assuming 'G3' is the final grade; passing is G3 >= 10
df['pass'] = df['pass'].astype(int)

# Features selection - use whichever are available in your CSV
features = ['G1', 'G2', 'studytime', 'failures', 'absences']
X = df[features]
y = df['pass']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# Streamlit UI
st.title("Student Pass Prediction App")
st.write(f"Model accuracy: {accuracy:.2f}")

with st.form("predict_form"):
    G1 = st.number_input("First period grade (G1)", 0, 20)
    G2 = st.number_input("Second period grade (G2)", 0, 20)
    studytime = st.number_input("Weekly study time (hours)", 0, 10)
    failures = st.number_input("Number of past class failures", 0, 5)
    absences = st.number_input("Number of absences", 0, 50)
    submitted = st.form_submit_button("Predict")

if submitted:
    X_new = pd.DataFrame([[G1, G2, studytime, failures, absences]], columns=features)
    prediction = model.predict(X_new)[0]
    st.success(f"Prediction: {'Pass' if prediction == 1 else 'Fail'}")

# Streamlit plots (without matplotlib)
st.subheader("Pass/Fail Distribution")
st.bar_chart(df['pass'].value_counts())

st.subheader("Average Grades by Pass/Fail")
st.bar_chart(df.groupby('pass')[['G1', 'G2', 'G3']].mean())
