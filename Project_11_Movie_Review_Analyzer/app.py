import streamlit as st
from transformers import pipeline
from pathlib import Path


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Movie Review Analyzer",
    page_icon="🎬",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background-color: #080808;
    color: white;
}


/* PAGE WIDTH */

.block-container {
    max-width: 1150px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}


/* TITLE */

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 5px;
}


.subtitle {
    text-align: center;
    color: #aaaaaa;
    font-size: 17px;
    margin-bottom: 35px;
}


/* SECTION TITLE */

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
}


/* RESULT CARD */

.result-card {
    background-color: #141414;
    border: 1px solid #333333;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
    margin-top: 25px;
}


.positive {
    color: #35d06f;
    font-size: 32px;
    font-weight: 800;
}


.negative {
    color: #ff4b4b;
    font-size: 32px;
    font-weight: 800;
}


.confidence {
    color: #bbbbbb;
    font-size: 18px;
    margin-top: 12px;
}


/* INFORMATION CARDS */

.info-card {
    background-color: #141414;
    border: 1px solid #292929;
    border-radius: 12px;
    padding: 22px;
    min-height: 155px;
}


.card-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 10px;
}


.card-text {
    color: #aaaaaa;
    font-size: 15px;
    line-height: 1.5;
}


/* BUTTON */

.stButton > button {
    width: 100%;
    background-color: #e50914;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px;
    font-size: 17px;
    font-weight: 700;
}


.stButton > button:hover {
    background-color: #ff1824;
    color: white;
    border: none;
}


/* TEXT AREA */

.stTextArea textarea {
    background-color: #151515;
    color: white;
    border: 1px solid #333333;
    border-radius: 10px;
}


/* HIDE STREAMLIT ELEMENTS */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource(show_spinner=False)
def load_model():

    return pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


# ==================================================
# HERO IMAGE
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "movie_review_cover.png"


if IMAGE_PATH.exists():

    st.image(
        str(IMAGE_PATH),
        width="stretch"
    )

else:

    st.error(
        "Cover image not found. Upload movie_review_cover.png "
        "in the same folder as app.py."
    )


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🎬 AI Movie Review Analyzer</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
<div class="subtitle">
Analyze Movie Reviews using Natural Language Processing and DistilBERT
</div>
""",
    unsafe_allow_html=True
)


# ==================================================
# LOAD MODEL
# ==================================================

with st.spinner("Loading AI model..."):

    classifier = load_model()


# ==================================================
# REVIEW INPUT
# ==================================================

st.markdown(
    '<div class="section-title">Analyze Movie Review</div>',
    unsafe_allow_html=True
)


review = st.text_area(

    "Movie Review",

    placeholder=(
        "Example: KGF Chapter 2 was an amazing movie "
        "with powerful action and excellent performances."
    ),

    height=150,

    label_visibility="collapsed"
)


analyze_button = st.button(
    "Analyze Sentiment",
    use_container_width=True
)


# ==================================================
# SENTIMENT ANALYSIS
# ==================================================

if analyze_button:

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

            result_class = "positive"

            emoji = "😊"

            result_text = "POSITIVE REVIEW"

        else:

            result_class = "negative"

            emoji = "😞"

            result_text = "NEGATIVE REVIEW"


        # IMPORTANT:
        # HTML starts without indentation to prevent
        # Streamlit rendering HTML as code.

        result_html = f"""
<div class="result-card">
<div class="{result_class}">{emoji} {result_text}</div>
<div class="confidence">Confidence Score: {confidence:.2f}%</div>
</div>
"""

        st.markdown(
            result_html,
            unsafe_allow_html=True
        )


# ==================================================
# HOW IT WORKS
# ==================================================

st.markdown("<br>", unsafe_allow_html=True)


st.markdown(
    '<div class="section-title">How It Works</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
<div class="info-card">
<div class="card-title">📝 Enter Review</div>
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
<div class="card-title">🤖 AI Analysis</div>
<div class="card-text">
DistilBERT analyzes the context and sentiment of the movie review.
</div>
</div>
""",
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
<div class="info-card">
<div class="card-title">📊 Prediction</div>
<div class="card-text">
The application displays Positive or Negative sentiment with a confidence score.
</div>
</div>
""",
        unsafe_allow_html=True
    )


# ==================================================
# FOOTER
# ==================================================

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
