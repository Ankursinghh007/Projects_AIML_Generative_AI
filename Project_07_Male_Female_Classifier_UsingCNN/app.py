import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

st.set_page_config(
    page_title="Gender Classifier Using CNN",
    page_icon="👤",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "gender_classifier.keras"

# Change this only if your training class mapping was different.
# Current mapping: Female = 0, Male = 1
CLASS_NAMES = ["Female", "Male"]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error("Unable to load the CNN model.")
    st.error(str(error))
    st.stop()


# Automatically get image size required by the CNN model
try:
    IMG_HEIGHT = int(model.input_shape[1])
    IMG_WIDTH = int(model.input_shape[2])
except Exception:
    IMG_HEIGHT = 64
    IMG_WIDTH = 64


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))

    image_array = np.array(image, dtype=np.float32)

    # Normalize pixels from 0-255 to 0-1
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_gender(image):
    processed_image = preprocess_image(image)

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    prediction = np.asarray(prediction)

    # Binary sigmoid output: [[0.85]]
    if prediction.size == 1:
        male_probability = float(prediction.reshape(-1)[0])

        male_probability = float(
            np.clip(male_probability, 0.0, 1.0)
        )

        female_probability = 1.0 - male_probability

        probabilities = np.array(
            [female_probability, male_probability]
        )

    # Two-class output: [[0.15, 0.85]]
    elif prediction.ndim >= 2 and prediction.shape[-1] == 2:
        probabilities = prediction[0].astype(float)

        # Apply softmax if model returns logits
        if (
            np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-3)
        ):
            probabilities = tf.nn.softmax(probabilities).numpy()

    else:
        raise ValueError(
            f"Unsupported model output shape: {prediction.shape}"
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    return predicted_class, confidence, probabilities


st.title("Gender Classifier Using CNN")

st.write(
    "Upload a face image and click Predict Gender to classify "
    "the image as Male or Female using a CNN model."
)

st.divider()


uploaded_file = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Invalid image file.")
        st.stop()

    st.image(
        image,
        caption="Uploaded Face Image",
        width=350
    )

    st.divider()

    if st.button(
        "Predict Gender",
        type="primary",
        use_container_width=True
    ):
        try:
            with st.spinner("Analyzing image using CNN model..."):
                predicted_class, confidence, probabilities = predict_gender(
                    image
                )

            female_probability = float(probabilities[0])
            male_probability = float(probabilities[1])

            st.subheader("Prediction Result")

            if predicted_class == "Male":
                st.success("♂️ Predicted Gender: Male")
            else:
                st.success("♀️ Predicted Gender: Female")

            st.metric(
                "Prediction Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.subheader("Prediction Probabilities")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "♀️ Female",
                    f"{female_probability * 100:.2f}%"
                )
                st.progress(female_probability)

            with col2:
                st.metric(
                    "♂️ Male",
                    f"{male_probability * 100:.2f}%"
                )
                st.progress(male_probability)

        except Exception as error:
            st.error("Prediction failed.")
            st.error(str(error))

else:
    st.info("Upload a face image to start gender classification.")


st.divider()

st.caption(
    "Gender Classifier Using CNN | TensorFlow | Keras | Streamlit"
)
