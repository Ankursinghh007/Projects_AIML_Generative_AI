```python
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Gender Classifier Using CNN",
    page_icon="👤",
    layout="centered"
)

# --------------------------------------------------
# MODEL PATH
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "gender_classifier.keras"

# IMPORTANT:
# Class order must match model training.
# Change to ["Male", "Female"] if your model used:
# Male = 0, Female = 1

CLASS_NAMES = ["Female", "Male"]


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "Model file not found. Make sure "
        "'gender_classifier.keras' is in the same folder as app.py."
    )
    st.stop()

except Exception as error:
    st.error("Unable to load the CNN model.")
    st.code(str(error))
    st.stop()


# --------------------------------------------------
# GET MODEL INPUT SIZE
# --------------------------------------------------

try:
    IMG_HEIGHT = int(model.input_shape[1])
    IMG_WIDTH = int(model.input_shape[2])

except Exception:
    IMG_HEIGHT = 64
    IMG_WIDTH = 64


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("👤 Gender Classifier Using CNN")

st.write(
    "Upload a face image and click **Predict Gender** "
    "to classify the image as **Male ♂️** or **Female ♀️** "
    "using a Convolutional Neural Network."
)

st.divider()


# --------------------------------------------------
# FILE UPLOADER
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Face Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (IMG_WIDTH, IMG_HEIGHT)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Normalize image pixels
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_gender(image):

    processed_image = preprocess_image(image)

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    prediction = np.asarray(prediction)


    # --------------------------------------------------
    # BINARY SIGMOID OUTPUT
    # Example: [[0.82]]
    # --------------------------------------------------

    if prediction.size == 1:

        male_probability = float(
            prediction.reshape(-1)[0]
        )

        male_probability = np.clip(
            male_probability,
            0.0,
            1.0
        )

        female_probability = 1.0 - male_probability

        probabilities = np.array([
            female_probability,
            male_probability
        ])


    # --------------------------------------------------
    # TWO CLASS SOFTMAX OUTPUT
    # Example: [[0.25, 0.75]]
    # --------------------------------------------------

    elif prediction.shape[-1] == 2:

        probabilities = prediction[0].astype(float)

        # Convert logits to probabilities if necessary
        if (
            np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.isclose(
                probabilities.sum(),
                1.0,
                atol=1e-3
            )
        ):

            probabilities = tf.nn.softmax(
                probabilities
            ).numpy()


    else:

        raise ValueError(
            f"Unsupported model output shape: {prediction.shape}"
        )


    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )


    return (
        predicted_class,
        confidence,
        probabilities
    )


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)

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
        "🔍 Predict Gender",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Analyzing image using CNN model..."
            ):

                (
                    predicted_class,
                    confidence,
                    probabilities

                ) = predict_gender(image)


            # --------------------------------------------------
            # PREDICTION RESULT
            # --------------------------------------------------

            st.subheader("🎯 Prediction Result")


            if predicted_class == "Male":

                st.success(
                    "♂️ Predicted Gender: Male"
                )

            else:

                st.success(
                    "♀️ Predicted Gender: Female"
                )


            st.metric(
                "Prediction Confidence",
                f"{confidence * 100:.2f}%"
            )


            # --------------------------------------------------
            # CLASS PROBABILITIES
            # --------------------------------------------------

            st.subheader("📊 Prediction Probabilities")


            female_probability = float(
                probabilities[0]
            )

            male_probability = float(
                probabilities[1]
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "♀️ Female",
                    f"{female_probability * 100:.2f}%"
                )

                st.progress(
                    float(female_probability)
                )


            with col2:

                st.metric(
                    "♂️ Male",
                    f"{male_probability * 100:.2f}%"
                )

                st.progress(
                    float(male_probability)
                )


        except Exception as error:

            st.error("Prediction failed.")

            st.code(str(error))


else:

    st.info(
        "📤 Upload a face image to start gender classification."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Gender Classifier Using CNN • TensorFlow • Keras • Streamlit"
)
```
