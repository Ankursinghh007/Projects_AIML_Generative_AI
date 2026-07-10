import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Data Outlier Analyzer",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    opacity: 0.75;
    margin-bottom: 25px;
}

.info-card {
    padding: 18px;
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 12px;
    margin-bottom: 20px;
}

div.stDownloadButton > button {
    width: 100%;
    border-radius: 8px;
    height: 45px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📈 Data Outlier Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Upload a CSV dataset, analyze numerical features, detect unusual
    observations using the IQR method, and export a cleaned dataset.
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# File Upload Section
# --------------------------------------------------

st.markdown("### 📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file to start analysis",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Upload a CSV dataset to explore numerical features and detect outliers."
    )

else:

    # --------------------------------------------------
    # Read Dataset
    # --------------------------------------------------

    try:
        df = pd.read_csv(uploaded_file)

    except Exception as e:
        st.error(f"Unable to read the CSV file: {e}")
        st.stop()

    if df.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if not numeric_columns:
        st.error("The dataset does not contain numerical columns.")
        st.stop()

    st.success("Dataset uploaded successfully.")

    # --------------------------------------------------
    # Dataset Overview
    # --------------------------------------------------

    st.markdown("### 📋 Dataset Overview")

    overview1, overview2, overview3 = st.columns(3)

    overview1.metric("Total Rows", df.shape[0])
    overview2.metric("Total Columns", df.shape[1])
    overview3.metric("Numerical Features", len(numeric_columns))

    with st.expander("View Dataset Preview", expanded=True):
        st.dataframe(
            df.head(10),
            use_container_width=True
        )

    # --------------------------------------------------
    # Feature Selection
    # --------------------------------------------------

    st.markdown("### 🔎 Select Feature for Analysis")

    selected_column = st.selectbox(
        "Choose a numerical column",
        numeric_columns
    )

    # Remove missing values for calculations

    column_data = df[selected_column].dropna()

    if column_data.empty:
        st.warning("The selected feature contains no valid numerical values.")
        st.stop()

    # --------------------------------------------------
    # IQR Calculation
    # --------------------------------------------------

    Q1 = column_data.quantile(0.25)
    Q3 = column_data.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    outlier_mask = (
        (df[selected_column] < lower_bound)
        | (df[selected_column] > upper_bound)
    )

    outliers = df[outlier_mask]

    cleaned_df = df[~outlier_mask].copy()

    outlier_percentage = (
        (len(outliers) / len(df)) * 100
        if len(df) > 0
        else 0
    )

    # --------------------------------------------------
    # Analysis Results
    # --------------------------------------------------

    st.markdown("### 📊 Analysis Results")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Lower Limit",
        f"{lower_bound:.2f}"
    )

    metric2.metric(
        "Upper Limit",
        f"{upper_bound:.2f}"
    )

    metric3.metric(
        "Outliers Found",
        len(outliers)
    )

    metric4.metric(
        "Outlier Percentage",
        f"{outlier_percentage:.2f}%"
    )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    st.markdown("### 📉 Data Visualization")

    tab1, tab2 = st.tabs(
        ["📦 Box Plot", "📊 Distribution"]
    )

    with tab1:

        fig, ax = plt.subplots(figsize=(10, 3))

        ax.boxplot(
            column_data,
            vert=False,
            patch_artist=True
        )

        ax.set_xlabel(selected_column)
        ax.set_title(f"Box Plot of {selected_column}")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    with tab2:

        fig2, ax2 = plt.subplots(figsize=(10, 4))

        ax2.hist(
            column_data,
            bins=30,
            edgecolor="black",
            alpha=0.8
        )

        ax2.axvline(
            lower_bound,
            linestyle="--",
            label="Lower Limit"
        )

        ax2.axvline(
            upper_bound,
            linestyle="--",
            label="Upper Limit"
        )

        ax2.set_xlabel(selected_column)
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"Distribution of {selected_column}")

        ax2.legend()

        plt.tight_layout()

        st.pyplot(fig2)

        plt.close(fig2)

    # --------------------------------------------------
    # Summary Statistics
    # --------------------------------------------------

    with st.expander("📌 View Summary Statistics"):

        statistics = column_data.describe().to_frame(
            name="Value"
        )

        st.dataframe(
            statistics,
            use_container_width=True
        )

    # --------------------------------------------------
    # Outlier Records
    # --------------------------------------------------

    st.markdown("### 🚨 Detected Outlier Records")

    if outliers.empty:

        st.success(
            "No outliers were detected for the selected feature."
        )

    else:

        st.dataframe(
            outliers,
            use_container_width=True
        )

    # --------------------------------------------------
    # Clean Dataset Section
    # --------------------------------------------------

    st.markdown("### 🧹 Cleaned Dataset")

    clean1, clean2, clean3 = st.columns(3)

    clean1.metric(
        "Original Rows",
        len(df)
    )

    clean2.metric(
        "Rows Removed",
        len(outliers)
    )

    clean3.metric(
        "Remaining Rows",
        len(cleaned_df)
    )

    with st.expander("Preview Cleaned Dataset"):

        st.dataframe(
            cleaned_df.head(20),
            use_container_width=True
        )

    # --------------------------------------------------
    # Download
    # --------------------------------------------------

    csv_data = cleaned_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Cleaned Dataset",
        data=csv_data,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )
