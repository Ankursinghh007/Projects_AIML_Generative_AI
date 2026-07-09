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

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #111827 50%,
        #1e293b 100%
    );
}

.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 35px;
}

.info-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 25px;
}

.prediction-card {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
}

.prediction-text {
    font-size: 2.2rem;
    font-weight: 700;
}

.confidence-text {
    color: #cbd5e1;
    font-size: 1.2rem;
    margin-top: 10px;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 50px;
    font-size: 0.9rem;
}

.stButton > button {
    width: 100%;
    height: 3rem;
    border-radius: 10px;
    font-weight: bold;
}

[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.04);
    padding: 20px;
    border-radius: 15px;
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
        MODEL_PATH,
        compile=False
    )


if not MODEL_PATH.exists():

    st.error(
        "model.keras file not found. "
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

    image = image.resize((299, 299))

    img_array = np.asarray(
        image,
        dtype=np.float32
    )

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return display_image, img_array


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_eye(img_array):

    result = model.predict(
        img_array,
        verbose=0
    )

    probability = float(result[0][0])


    # CLASS MAPPING
    #
    # Female = 0
    # Male   = 1


    if probability >= 0.5:

        predicted_class = "♂️ 👁️ Male Eye"

        confidence = probability

    else:

        predicted_class = "♀️ 👁️ Female Eye"

        confidence = 1 - probability


    return predicted_class, confidence, probability


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("👁️ Eye Classifier")

    st.markdown("---")


    st.subheader("About")

    st.write(
        """
        This application uses a Convolutional Neural Network
        (CNN) to classify an uploaded eye image into:

        ♂️ 👁️ Male Eye

        ♀️ 👁️ Female Eye
        """
    )


    st.markdown("---")


    st.subheader("How to Use")

    st.write(
        """
        1. Upload a clear eye image.

        2. Use JPG, JPEG, or PNG format.

        3. Click Predict Eye Class.

        4. View the prediction and confidence score.
        """
    )


    st.markdown("---")

    st.caption(
        "CNN-Based Eye Image Classification Project"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">

        👁️ Male & Female Eye Classifier

    </div>


    <div class="subtitle">

        CNN-Based Eye Image Classification System

    </div>
    """,

    unsafe_allow_html=True
)


# ============================================================
# INFORMATION CARD
# ============================================================

st.markdown(
    """
    <div class="info-card">

        <h3>Upload an Eye Image</h3>

        <p>

        Upload a clear eye image and let the trained
        Convolutional Neural Network analyze the image
        and classify it as a Male Eye or Female Eye.

        </p>

    </div>
    """,

    unsafe_allow_html=True
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(

    "Choose an Eye Image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# IMAGE + PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        display_image, processed_image = preprocess_image(
            uploaded_file
        )


        left_column, right_column = st.columns(
            [1, 1],
            gap="large"
        )


        # IMAGE COLUMN

        with left_column:

            st.subheader("Uploaded Eye Image")

            st.image(
                display_image,
                use_container_width=True
            )


        # PREDICTION COLUMN

        with right_column:

            st.subheader("Prediction Result")


            if st.button(
                "🔍 Predict Eye Class",
                type="primary"
            ):


                with st.spinner(
                    "Analyzing the eye image..."
                ):


                    predicted_class, confidence, raw_probability = (
                        predict_eye(processed_image)
                    )


                st.markdown(
                    f"""
                    <div class="prediction-card">

                        <div class="prediction-text">

                            {predicted_class}

                        </div>


                        <div class="confidence-text">

                            Confidence: {confidence * 100:.2f}%

                        </div>

                    </div>
                    """,

                    unsafe_allow_html=True
                )


                st.progress(
                    float(confidence)
                )


                st.markdown(
                    "### Model Output Details"
                )


                female_column, male_column = st.columns(2)


                with female_column:

                    st.metric(

                        "♀️ Female Probability",

                        f"{(1 - raw_probability) * 100:.2f}%"
                    )


                with male_column:

                    st.metric(

                        "♂️ Male Probability",

                        f"{raw_probability * 100:.2f}%"
                    )


    except Exception as error:

        st.error(
            "Unable to process the uploaded image. "
            "Upload a valid JPG, JPEG, or PNG image."
        )

        st.exception(error)


else:

    st.info(
        "Upload an eye image to start classification."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")


st.subheader("Model Information")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Model Type",
        "CNN"
    )


with col2:

    st.metric(
        "Input Size",
        "299 × 299"
    )


with col3:

    st.metric(
        "Classes",
        "2"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        Eye Gender Classification System • Streamlit • TensorFlow

    </div>
    """,

    unsafe_allow_html=True
)
