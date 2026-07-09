import streamlit as st
import numpy as np
from PIL import Image
import joblib

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Face Gender Classifier",
    page_icon="♂️♀️",
    layout="centered"
)

# -----------------------------------
# Load Model
# -----------------------------------
@st.cache_resource
def load_model():
    return joblib.load("male_female_model.pkl")

try:
    model = load_model()
except Exception:
    st.error("❌ Unable to load the trained model.")
    st.stop()

IMG_SIZE = 64

# -----------------------------------
# Header
# -----------------------------------
st.title("♂️ ♀️ Face Gender Classifier")
st.write(
    "Upload a face image and click **Predict** to classify it as **Male** or **Female**."
)

st.divider()

# -----------------------------------
# Upload Image
# -----------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# Prediction
# -----------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    st.divider()

    if st.button("🔍 Predict", use_container_width=True):

        with st.spinner("Analyzing image..."):

            # Image Preprocessing
            img = image.resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img)
            img = img.flatten()

            # Prediction
            prediction = model.predict([img])[0]
            probability = model.predict_proba([img])[0]

        # -----------------------------------
        # Prediction Result
        # -----------------------------------
        st.subheader("Prediction Result")

        # Reverse Mapping (0 = Male, 1 = Female)
        if prediction == 0:
            st.success("👨‍💼 Male")
        else:
            st.success("👩‍💼 Female")

        # -----------------------------------
        # Prediction Confidence
        # -----------------------------------
        st.subheader("Prediction Confidence")

        male_prob = probability[0]
        female_prob = probability[1]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "👨‍💼 Male",
                f"{male_prob * 100:.2f}%"
            )
            st.progress(float(male_prob))

        with col2:
            st.metric(
                "👩‍💼 Female",
                f"{female_prob * 100:.2f}%"
            )
            st.progress(float(female_prob))

else:
    st.info("👆 Please upload a face image to begin.")
