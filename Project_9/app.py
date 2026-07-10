import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Male Female Eye Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Remove extra top spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}


/* Main App Background */
.stApp {
    background:
        radial-gradient(circle at top left, #172554 0%, transparent 30%),
        radial-gradient(circle at bottom right, #312e81 0%, transparent 25%),
        linear-gradient(135deg, #020617 0%, #0f172a 55%, #111827 100%);
}


/* Header Container */
.hero-container {
    text-align: center;
    padding: 30px 20px;
    margin-bottom: 25px;
}


/* Eye Logo */
.eye-logo {
    font-size: 3.8rem;
    margin-bottom: 5px;
}


/* Main Heading */
.main-title {
    font-size: clamp(2.1rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 10px;
}


/* Subtitle */
.subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
}


/* Intro Card */
.info-card {
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 25px;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.20);
}


/* Intro Heading */
.info-heading {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 8px;
}


/* Intro Text */
.info-text {
    color: #cbd5e1;
    font-size: 1rem;
    line-height: 1.7;
}


/* Result Card */
.result-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.20);
    border-radius: 20px;
    padding: 32px 20px;
    text-align: center;
    margin-top: 15px;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25);
}


/* Result Icon */
.result-icon {
    font-size: 3.2rem;
}


/* Result Label */
.result-label {
    font-size: 2rem;
    font-weight: 800;
    margin-top: 8px;
}


/* Confidence */
.confidence-text {
    color: #cbd5e1;
    font-size: 1.1rem;
    margin-top: 10px;
}


/* Section Heading */
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 12px;
}


/* Upload Area */
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 18px;
    padding: 15px;
}


/* Buttons */
.stButton > button {
    width: 100%;
    min-height: 3.2rem;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 700;
}


/* Metrics */
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.70);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 15px;
    padding: 18px;
}


/* Sidebar */
[data-testid="stSidebar"] {
    background: #080f1e;
}


/* Horizontal Line */
hr {
    border-color: rgba(148, 163, 184, 0.15);
}


/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.9rem;
    padding-top: 30px;
    padding-bottom: 10px;
}


/* Mobile Optimization */
@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero-container {
        padding: 15px 5px;
    }

    .eye-logo {
        font-size: 3rem;
    }

    .main-title {
        font-size: 2.2rem;
    }

    .info-card {
        padding: 20px;
    }

    .result-label {
        font-size: 1.7rem;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.keras"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_eye_model():

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


if not MODEL_PATH.exists():

    st.error(
        "model.keras file was not found. "
        "Place model.keras in the same folder as app.py."
    )

    st.stop()


try:

    model = load_eye_model()

except Exception as error:

    st.error("Unable to load model.keras.")

    st.exception(error)

    st.stop()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(uploaded_file):

    image = Image.open(uploaded_file).convert("RGB")

    display_image = image.copy()

    resized_image = image.resize((299, 299))

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return display_image, image_array


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_eye(image_array):

    result = model.predict(
        image_array,
        verbose=0
    )

    male_probability = float(result[0][0])

    female_probability = 1.0 - male_probability


    # Class Mapping:
    # Female = 0
    # Male = 1


    if male_probability >= 0.5:

        predicted_class = "Male Eye"

        result_icon = "♂️ 👁️"

        confidence = male_probability

    else:

        predicted_class = "Female Eye"

        result_icon = "♀️ 👁️"

        confidence = female_probability


    return (
        predicted_class,
        result_icon,
        confidence,
        female_probability,
        male_probability
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 👁️ Eye Detection")

    st.caption("CNN Image Classification Project")

    st.divider()


    st.markdown("### About")

    st.write(
        "This application uses a trained Convolutional "
        "Neural Network to classify an uploaded eye image "
        "as a Male Eye or Female Eye."
    )


    st.divider()


    st.markdown("### Classes")

    st.markdown("**♂️ 👁️ Male Eye**")

    st.markdown("**♀️ 👁️ Female Eye**")


    st.divider()


    st.markdown("### How to Use")

    st.write("1. Upload a clear eye image.")

    st.write("2. Click the Predict Eye Class button.")

    st.write("3. View the predicted class.")

    st.write("4. Check confidence and probabilities.")


    st.divider()


    st.markdown("### Supported Formats")

    st.write("JPG • JPEG • PNG")


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero-container">
    <div class="eye-logo">👁️</div>
    <div class="main-title">Male Female Eye Detection</div>
    <div class="subtitle">AI-Powered Eye Image Classification using Convolutional Neural Networks</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# INFORMATION CARD
# ============================================================

st.markdown("""
<div class="info-card">
    <div class="info-heading">Upload an Eye Image</div>
    <div class="info-text">
        Upload a clear eye image and the trained CNN model
        will analyze its visual features and classify it
        as a Male Eye or Female Eye.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Choose an Eye Image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    try:

        display_image, processed_image = preprocess_image(
            uploaded_file
        )


        image_column, result_column = st.columns(
            [1, 1],
            gap="large"
        )


        # ====================================================
        # IMAGE COLUMN
        # ====================================================

        with image_column:

            st.markdown(
                '<div class="section-title">Uploaded Image</div>',
                unsafe_allow_html=True
            )


            # Controlled image width

            image_space_left, image_space_center, image_space_right = (
                st.columns([0.08, 0.84, 0.08])
            )


            with image_space_center:

                st.image(
                    display_image,
                    use_container_width=True
                )


        # ====================================================
        # RESULT COLUMN
        # ====================================================

        with result_column:

            st.markdown(
                '<div class="section-title">Prediction Result</div>',
                unsafe_allow_html=True
            )


            predict_button = st.button(
                "🔍 Predict Eye Class",
                type="primary"
            )


            if predict_button:

                with st.spinner(
                    "Analyzing the uploaded eye image..."
                ):

                    (
                        predicted_class,
                        result_icon,
                        confidence,
                        female_probability,
                        male_probability

                    ) = predict_eye(processed_image)


                st.markdown(
                    f"""
<div class="result-card">
    <div class="result-icon">{result_icon}</div>
    <div class="result-label">{predicted_class}</div>
    <div class="confidence-text">Confidence: {confidence * 100:.2f}%</div>
</div>
""",
                    unsafe_allow_html=True
                )


                st.write("")

                st.progress(float(confidence))


                st.markdown("### Prediction Probabilities")


                probability_col1, probability_col2 = st.columns(2)


                with probability_col1:

                    st.metric(
                        "♀️ Female Eye",
                        f"{female_probability * 100:.2f}%"
                    )


                with probability_col2:

                    st.metric(
                        "♂️ Male Eye",
                        f"{male_probability * 100:.2f}%"
                    )


            else:

                st.info(
                    "Click Predict Eye Class to analyze "
                    "the uploaded image."
                )


    except Exception as error:

        st.error(
            "The uploaded image could not be processed."
        )

        st.exception(error)


else:

    st.info(
        "Upload a JPG, JPEG, or PNG eye image to start prediction."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.markdown("## Model Information")


model_col1, model_col2, model_col3 = st.columns(3)


with model_col1:

    st.metric(
        "Architecture",
        "CNN"
    )


with model_col2:

    st.metric(
        "Input Resolution",
        "299 × 299"
    )


with model_col3:

    st.metric(
        "Output Classes",
        "2"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    Male Female Eye Detection • Built with Streamlit and TensorFlow
</div>
""", unsafe_allow_html=True)
