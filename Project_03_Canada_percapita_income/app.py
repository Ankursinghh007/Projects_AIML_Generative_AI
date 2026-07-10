import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path


# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Canada Per Capita Income Prediction",
    page_icon="🇨🇦",
    layout="centered"
)


# -----------------------------------
# App Title
# -----------------------------------
st.title("🇨🇦 Canada Per Capita Income Prediction")

st.write(
    "Predict Canada's Per Capita Income for a given year "
    "using Linear Regression."
)


# -----------------------------------
# Get Current Folder Path
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent


# -----------------------------------
# Load Dataset
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        BASE_DIR / "canada_per_capita_income.csv"
    )


df = load_data()


# -----------------------------------
# Display Dataset
# -----------------------------------
st.subheader("Canada Per Capita Income Dataset")

st.dataframe(
    df,
    use_container_width=True
)


# -----------------------------------
# Prepare Dataset
# -----------------------------------
X = df[["year"]]

y = df["per capita income (US$)"]


# -----------------------------------
# Train Linear Regression Model
# -----------------------------------
model = LinearRegression()

model.fit(X, y)


# -----------------------------------
# Data Visualization
# -----------------------------------
st.subheader("Per Capita Income Trend")

st.line_chart(
    df.set_index("year")["per capita income (US$)"]
)


# -----------------------------------
# User Input
# -----------------------------------
st.subheader("Enter Year")

year = st.number_input(
    "Year",
    min_value=1970,
    max_value=2100,
    value=2026,
    step=1
)


# -----------------------------------
# Prediction
# -----------------------------------
if st.button("Predict Per Capita Income"):

    input_data = pd.DataFrame(
        {"year": [year]}
    )

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Per Capita Income for {year}: "
        f"${prediction:,.2f}"
    )


# -----------------------------------
# Model Information
# -----------------------------------
st.subheader("Model Details")

st.write(
    "Coefficient:",
    round(model.coef_[0], 4)
)

st.write(
    "Intercept:",
    round(model.intercept_, 4)
)

st.write(
    "Model Score (R²):",
    round(model.score(X, y), 4)
)
