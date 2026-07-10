
import streamlit as st
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌺",
    layout="centered"
)


@st.cache_resource
def build_model():
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.25,
        random_state=42,
        stratify=iris.target
    )

    knn_model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )

    knn_model.fit(X_train, y_train)

    model_accuracy = knn_model.score(X_test, y_test)

    return knn_model, iris.target_names, model_accuracy


model, species_names, accuracy = build_model()


st.title("🌺 Iris Species Classifier")

st.write(
    "Enter the flower measurements below to identify the most likely "
    "Iris species using a K-Nearest Neighbors classification model."
)

st.metric(
    label="Model Test Accuracy",
    value=f"{accuracy * 100:.2f}%"
)

st.divider()


st.subheader("Flower Measurements")

st.caption("Adjust the measurements in centimeters.")

col1, col2 = st.columns(2)


with col1:

    sepal_length = st.slider(
        "Sepal Length",
        min_value=4.0,
        max_value=8.0,
        value=5.1,
        step=0.1,
        help="Length of the flower's sepal in centimeters."
    )

    petal_length = st.slider(
        "Petal Length",
        min_value=1.0,
        max_value=7.0,
        value=1.4,
        step=0.1,
        help="Length of the flower's petal in centimeters."
    )


with col2:

    sepal_width = st.slider(
        "Sepal Width",
        min_value=2.0,
        max_value=5.0,
        value=3.5,
        step=0.1,
        help="Width of the flower's sepal in centimeters."
    )

    petal_width = st.slider(
        "Petal Width",
        min_value=0.1,
        max_value=3.0,
        value=0.2,
        step=0.1,
        help="Width of the flower's petal in centimeters."
    )


st.divider()


if st.button(
    "Predict Iris Species",
    type="primary",
    use_container_width=True
):

    input_data = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    predicted_index = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    predicted_species = species_names[predicted_index].title()

    confidence = float(probabilities[predicted_index])


    st.subheader("Prediction Result")

    st.success(
        f"🌸 Predicted Species: {predicted_species}"
    )

    st.metric(
        label="Prediction Confidence",
        value=f"{confidence * 100:.2f}%"
    )


    st.subheader("Species Probabilities")


    probability_columns = st.columns(len(species_names))


    for index, species in enumerate(species_names):

        probability = float(probabilities[index])

        with probability_columns[index]:

            st.metric(
                species.title(),
                f"{probability * 100:.2f}%"
            )

            st.progress(probability)


st.divider()


with st.expander("About the Model"):

    st.write(
        "This application uses the K-Nearest Neighbors (KNN) "
        "classification algorithm to predict Iris flower species."
    )

    st.write(
        "The model is trained using the Iris dataset available "
        "in Scikit-learn."
    )

    st.write(
        "The prediction is based on four numerical features: "
        "sepal length, sepal width, petal length, and petal width."
    )


st.caption(
    "Iris Species Classification | Machine Learning | KNN | Streamlit"
)
