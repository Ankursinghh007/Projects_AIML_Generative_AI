
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Employee Retention Prediction", page_icon="👨‍💼", layout="centered")

st.title("👨‍💼 Employee Retention Prediction")
st.write("Predict whether an employee is likely to leave the company using Logistic Regression.")

@st.cache_data
def load_data():
    return pd.read_csv("HR_comma_sep.csv")

df = load_data()

# Same features used in logistic_regression_exercise.py
features = ["satisfaction_level", "average_montly_hours", "promotion_last_5years", "salary"]
subdf = df[features].copy()

# Convert salary categorical values to dummy variables
X = pd.get_dummies(subdf, columns=["salary"], prefix="salary")
y = df["left"]

# Keep split reproducible and use the same 30% training size as the exercise file
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

st.subheader("Enter Employee Details")

satisfaction_level = st.slider(
    "Satisfaction Level",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

average_montly_hours = st.number_input(
    "Average Monthly Hours",
    min_value=1,
    max_value=500,
    value=200,
    step=1
)

promotion_last_5years = st.selectbox(
    "Promotion in Last 5 Years",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

salary = st.selectbox("Salary Level", options=["low", "medium", "high"])

if st.button("Predict Employee Retention"):
    input_data = pd.DataFrame([{
        "satisfaction_level": satisfaction_level,
        "average_montly_hours": average_montly_hours,
        "promotion_last_5years": promotion_last_5years,
        "salary": salary
    }])

    input_data = pd.get_dummies(input_data, columns=["salary"], prefix="salary")
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_data)[0]
    leave_probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("Prediction: Employee is likely to leave the company.")
    else:
        st.success("Prediction: Employee is likely to stay in the company.")

    st.write(f"Probability of Leaving: {leave_probability:.2%}")

st.subheader("Model Details")
st.write(f"Model Accuracy: {model.score(X_test, y_test):.2%}")
st.write("Algorithm: Logistic Regression")
