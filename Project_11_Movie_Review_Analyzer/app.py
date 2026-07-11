import streamlit as st
from transformers import pipeline
from PIL import Image
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Review Analyzer",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# LOAD SENTIMENT MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True
    )


classifier = load_model()


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: #080808;
    color: white;
}

.block-container {
    max-width: 1100px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}


/* HERO IMAGE */

.hero-image {
    border-radius: 14px;
}


/* MAIN TITLE */

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #aaaaaa;
    font-size: 17px;
    margin-bottom: 35px;
}


/* INPUT SECTION */

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-bottom: 10px;
}


/* RESULT CARD */

.result-card {
    background: #151515;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #303030;
    text-align: center;
    margin-top: 25px;
}

.positive {
    color: #35d06f;
    font-size: 35px;
    font-weight: 800;
}

.negative {
    color: #ff4b4b;
    font-size: 35px;
    font-weight: 800;
}

.confidence {
    color: #bbbbbb;
    font-size: 18px;
}


/* INFO CARDS */

.info-card {
    background: #121212;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #292929;
    height: 150px;
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}

.card-text {
    color: #aaaaaa;
    font-size: 15px;
}


/* BUTTON */

.stButton > button {

    width: 100%;

    background: #e50914;

    color: white;

    border: none;

    border-radius: 8px;

    padding: 12px;

    font-size: 17px;

    font-weight: 700;
}

.stButton > button:hover {

    background: #ff1824;

    color: white;
}


/* TEXT AREA */

.stTextArea textarea {

    background: #151515;

    color: white;

    border-radius: 10px;

    border: 1px solid #333333;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HERO IMAGE
# --------------------------------------------------

IMAGE_PATH = "movie_review_cover.png"

if os.path.exists(IMAGE_PATH):

    image = Image.open(IMAGE_PATH)

    st.image(
        image,
        use_container_width=True
    )

else:

    st.warning(
        "Cover image not found. Add movie_review_cover.png to the project folder."
    )


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎬 AI Movie Review Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Analyze movie reviews using Natural Language Processing and DistilBERT
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# REVIEW INPUT
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Analyze Movie Review</div>',
    unsafe_allow_html=True
)


review = st.text_area(

    "Enter your movie review",

    placeholder="Example: KGF Chapter 2 was an amazing movie with powerful action and performances.",

    height=150,

    label_visibility="collapsed"
)


analyze = st.button(
    "Analyze Sentiment",
    use_container_width=True
)


# --------------------------------------------------
# SENTIMENT ANALYSIS
# --------------------------------------------------

if analyze:

    if not review.strip():

        st.warning("Please enter a movie review.")

    else:

        with st.spinner("Analyzing review..."):

            result = classifier(
                review,
                truncation=True,
                max_length=512
            )[0]


        sentiment = result["label"]

        confidence = result["score"] * 100


        if sentiment == "POSITIVE":

            st.markdown(

                f"""
                <div class="result-card">

                    <div class="positive">
                    😊 POSITIVE REVIEW
                    </div>

                    <br>

                    <div class="confidence">
                    Confidence Score: {confidence:.2f}%
                    </div>

                </div>
                """,

                unsafe_allow_html=True
            )


        else:

            st.markdown(

                f"""
                <div class="result-card">

                    <div class="negative">
                    😞 NEGATIVE REVIEW
                    </div>

                    <br>

                    <div class="confidence">
                    Confidence Score: {confidence:.2f}%
                    </div>

                </div>
                """,

                unsafe_allow_html=True
            )


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">How It Works</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        <div class="info-card">

        <div class="card-title">
        📝 User Review
        </div>

        <div class="card-text">
        Enter any English movie review that you want to analyze.
        </div>

        </div>
        """,

        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="info-card">

        <div class="card-title">
        🤖 DistilBERT Model
        </div>

        <div class="card-text">
        The transformer model analyzes the context and sentiment of the review.
        </div>

        </div>
        """,

        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="info-card">

        <div class="card-title">
        📊 Prediction
        </div>

        <div class="card-text">
        The application displays Positive or Negative sentiment with confidence score.
        </div>

        </div>
        """,

        unsafe_allow_html=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="
        text-align:center;
        color:#777777;
        border-top:1px solid #222222;
        padding-top:20px;
    ">

    Built with Python • Streamlit • Hugging Face Transformers • DistilBERT

    </div>
    """,

    unsafe_allow_html=True
)
