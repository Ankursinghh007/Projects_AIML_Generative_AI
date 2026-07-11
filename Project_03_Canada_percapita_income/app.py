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

DATA_PATH = BASE_DIR / "canada_per_capita_income.csv"


# -----------------------------------
# Load Dataset
# -----------------------------------
@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


# -----------------------------------
# Load Dataset Safely
# -----------------------------------
try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "Dataset not found. Make sure "
        "'canada_per_capita_income.csv' is in the same folder as app.py."
    )

    st.stop()

except Exception as error:

    st.error(f"Error loading dataset: {error}")

    st.stop()


# -----------------------------------
# Validate Dataset Columns
# -----------------------------------
required_columns = [
    "year",
    "per capita income (US$)"
]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    st.error(
        f"Missing columns in dataset: {missing_columns}"
    )

    st.write(
        "Available columns:",
        df.columns.tolist()
    )

    st.stop()


# -----------------------------------
# Clean Dataset
# -----------------------------------
df = df.dropna(
    subset=required_columns
).copy()


df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce"
)


df["per capita income (US$)"] = pd.to_numeric(
    df["per capita income (US$)"],
    errors="coerce"
)


df = df.dropna(
    subset=required_columns
)


# -----------------------------------
# Display Dataset
# -----------------------------------
st.subheader("Canada Per Capita Income Dataset")

st.dataframe(
    df,
    width="stretch",
    hide_index=True
)


# -----------------------------------
# Prepare Dataset
# -----------------------------------
X = df[["year"]]

y = df["per capita income (US$)"]


# -----------------------------------
# Train Linear Regression Model
# -----------------------------------
@st.cache_resource
def train_model(features, target):

    linear_model = LinearRegression()

    linear_model.fit(features, target)

    return linear_model


model = train_model(X, y)


# -----------------------------------
# Data Visualization
# -----------------------------------
st.subheader("Per Capita Income Trend")


chart_data = (

    df

    .sort_values("year")

    .set_index("year")[["per capita income (US$)"]]

)


st.line_chart(
    chart_data
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
if st.button(
    "Predict Per Capita Income",
    width="stretch"
):

    input_data = pd.DataFrame({

        "year": [year]

    })


    prediction = model.predict(input_data)[0]


    st.success(

        f"Predicted Per Capita Income for {year}: "
        f"${prediction:,.2f}"

    )


# -----------------------------------
# Model Information
# -----------------------------------
st.subheader("Model Details")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(

        "Coefficient",

        f"{model.coef_[0]:,.2f}"

    )


with col2:

    st.metric(

        "Intercept",

        f"{model.intercept_:,.2f}"

    )


with col3:

    st.metric(

        "R² Score",

        f"{model.score(X, y):.4f}"

    )
