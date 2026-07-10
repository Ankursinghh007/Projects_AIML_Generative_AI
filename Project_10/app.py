import streamlit as st
from google import genai
from openai import OpenAI
from bs4 import BeautifulSoup
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Star Health Insurance Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.stApp {
    background:
        radial-gradient(circle at top left, #123b52 0%, transparent 30%),
        radial-gradient(circle at bottom right, #14532d 0%, transparent 25%),
        linear-gradient(135deg, #020617 0%, #0f172a 55%, #111827 100%);
}

.hero {
    text-align: center;
    padding: 25px 15px 35px 15px;
}

.hero-icon {
    font-size: 4rem;
}

.hero-title {
    font-size: clamp(2.3rem, 5vw, 3.8rem);
    font-weight: 800;
    margin-top: 10px;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.08rem;
    margin-top: 10px;
}

.info-card {
    background: rgba(15, 23, 42, 0.82);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 22px;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.22);
}

.info-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.info-text {
    color: #cbd5e1;
    line-height: 1.7;
}

.suggestion-title {
    color: #cbd5e1;
    font-size: 1.05rem;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 10px;
}

[data-testid="stSidebar"] {
    background: #06111c;
}

[data-testid="stChatMessage"] {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 17px;
    padding: 10px;
    margin-bottom: 10px;
}

.stButton > button {
    width: 100%;
    min-height: 3rem;
    border-radius: 12px;
    font-weight: 600;
}

[data-testid="stTextInput"] input {
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.9rem;
    padding-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD STAR HEALTH HTML
# ============================================================

FILE_PATH = Path(__file__).resolve().parent / "starhealth.html"


@st.cache_data
def load_document():

    if not FILE_PATH.exists():
        return []

    with open(
        FILE_PATH,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        soup = BeautifulSoup(
            file.read(),
            "html.parser"
        )

    # Remove website noise

    for tag in soup([
        "script",
        "style",
        "svg",
        "nav",
        "header",
        "footer",
        "iframe",
        "noscript",
        "form"
    ]):

        tag.decompose()

    content = (
        soup.find("article")
        or soup.find("main")
        or soup.body
    )

    if content is None:
        return []

    paragraphs = []

    for tag in content.find_all(
        ["h1", "h2", "h3", "p", "li"]
    ):

        text = re.sub(
            r"\s+",
            " ",
            tag.get_text(" ", strip=True)
        )

        if len(text) >= 30:
            paragraphs.append(text)

    return paragraphs


paragraphs = load_document()


if not paragraphs:

    st.error(
        "Unable to load starhealth.html. "
        "Place it in the same folder as app.py."
    )

    st.stop()


# ============================================================
# CREATE CHUNKS
# ============================================================

@st.cache_data
def create_chunks(
    paragraphs,
    max_size=1800
):

    chunks = []

    current_chunk = ""

    for paragraph in paragraphs:

        if (
            current_chunk
            and len(current_chunk) + len(paragraph) > max_size
        ):

            chunks.append(current_chunk)

            current_chunk = paragraph

        else:

            current_chunk += " " + paragraph

    if current_chunk:

        chunks.append(current_chunk)

    return chunks


chunks = create_chunks(paragraphs)


# ============================================================
# TF-IDF INDEX
# ============================================================

@st.cache_resource
def create_index(chunks):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    vectors = vectorizer.fit_transform(chunks)

    return vectorizer, vectors


vectorizer, vectors = create_index(chunks)


# ============================================================
# RETRIEVE INFORMATION
# ============================================================

def retrieve_context(
    question,
    top_k=5
):

    question_vector = vectorizer.transform(
        [question]
    )

    scores = cosine_similarity(
        question_vector,
        vectors
    )[0]

    ranked_indexes = scores.argsort()[::-1]

    if scores[ranked_indexes[0]] < 0.03:

        return ""

    selected_chunks = []

    for index in ranked_indexes:

        if scores[index] > 0:

            selected_chunks.append(
                chunks[index]
            )

        if len(selected_chunks) == top_k:
            break

    return "\n\n---\n\n".join(
        selected_chunks
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    provider,
    api_key
):

    context = retrieve_context(question)

    if not context:

        return (
            "I could not find this information in the "
            "available health insurance knowledge base."
        )

    history = "\n".join(

        f"{message['role']}: {message['content']}"

        for message in st.session_state.messages[-6:]
    )

    prompt = f"""
You are a Star Health Insurance Assistant.

Answer the current question using ONLY the provided
health insurance information.

RULES:

1. Do not use outside knowledge.

2. Do not invent policy names, benefits, premiums,
coverage details, exclusions or eligibility rules.

3. Conversation history may only be used to understand
follow-up questions.

4. If the answer is unavailable, clearly say:
"I could not find this information in the available health insurance knowledge base."

5. Give direct and easy-to-understand answers.

6. Use bullet points when useful.

7. Do not mention APIs, prompts, documents,
retrieval systems or internal processing.

HEALTH INSURANCE INFORMATION:

{context}

CONVERSATION HISTORY:

{history}

CURRENT QUESTION:

{question}

ANSWER:
"""

    if provider == "Gemini":

        client = genai.Client(
            api_key=api_key
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text.strip()

    else:

        client = OpenAI(
            api_key=api_key
        )

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        return response.output_text.strip()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "assistant_active" not in st.session_state:
    st.session_state.assistant_active = False

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "provider" not in st.session_state:
    st.session_state.provider = ""

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏥 Insurance Assistant")

    st.caption(
        "Context-Aware Health Insurance Support"
    )

    st.divider()

    st.subheader("Assistant Status")

    if st.session_state.assistant_active:

        st.success("Assistant Active")

    else:

        st.warning("Assistant Not Activated")

    st.divider()

    st.subheader("Explore Topics")

    st.write("🛡️ Insurance Plans")
    st.write("👨‍👩‍👧‍👦 Family Insurance")
    st.write("🤰 Maternity Insurance")
    st.write("👴 Senior Citizen Plans")
    st.write("🏥 Policy Benefits")

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.pending_question = None

        st.rerun()


    if st.session_state.assistant_active:

        if st.button(
            "🔒 Deactivate Assistant",
            use_container_width=True
        ):

            st.session_state.assistant_active = False

            st.session_state.api_key = ""

            st.session_state.provider = ""

            st.session_state.messages = []

            st.session_state.pending_question = None

            st.rerun()


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    '<div class="hero"><div class="hero-icon">🏥</div><div class="hero-title">Star Health Insurance Assistant</div><div class="hero-subtitle">Explore Health Insurance Plans, Policy Benefits and Insurance Information</div></div>',
    unsafe_allow_html=True
)


# ============================================================
# ACTIVATION SECTION
# ============================================================

if not st.session_state.assistant_active:

    st.markdown(
        '<div class="info-card"><div class="info-title">🔑 Activate Insurance Assistant</div><div class="info-text">Select an API provider, enter the corresponding API key, and activate the assistant to start asking health insurance questions.</div></div>',
        unsafe_allow_html=True
    )

    provider = st.selectbox(
        "Select API Provider",
        ["Gemini", "OpenAI"]
    )

    entered_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Enter your API key..."
    )

    if st.button(
        "Activate Assistant",
        type="primary",
        use_container_width=True
    ):

        if not entered_key.strip():

            st.warning(
                "Please enter an API key."
            )

        else:

            st.session_state.api_key = (
                entered_key.strip()
            )

            st.session_state.provider = provider

            st.session_state.assistant_active = True

            st.success(
                "Assistant activated successfully."
            )

            st.rerun()


# ============================================================
# CHAT SECTION
# ============================================================

else:

    st.markdown(
        '<div class="info-card"><div class="info-title">💬 How can I help you?</div><div class="info-text">Ask questions about health insurance plans, policy benefits, maternity insurance, suitable beneficiaries and other information available in the knowledge base.</div></div>',
        unsafe_allow_html=True
    )


    # DISPLAY CHAT HISTORY

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # ========================================================
    # QUESTION INPUT
    # ========================================================

    typed_question = st.chat_input(
        "Ask a health insurance question..."
    )


    # ========================================================
    # SUGGESTED QUESTIONS
    # ========================================================

    st.markdown(
        '<div class="suggestion-title">Suggested Questions</div>',
        unsafe_allow_html=True
    )


    suggestions = [

        "What are the different types of health insurance plans?",

        "What benefits do health insurance policies offer?",

        "Why should you get health insurance when you are young?",

        "What are maternity health insurance plans?",

        "What are the different health insurance schemes in India?"

    ]


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            suggestions[0],
            key="suggestion1",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                suggestions[0]
            )

            st.rerun()


        if st.button(
            suggestions[2],
            key="suggestion3",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                suggestions[2]
            )

            st.rerun()


    with col2:

        if st.button(
            suggestions[1],
            key="suggestion2",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                suggestions[1]
            )

            st.rerun()


        if st.button(
            suggestions[3],
            key="suggestion4",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                suggestions[3]
            )

            st.rerun()


    if st.button(
        suggestions[4],
        key="suggestion5",
        use_container_width=True
    ):

        st.session_state.pending_question = (
            suggestions[4]
        )

        st.rerun()


    # ========================================================
    # SELECT QUESTION
    # ========================================================

    question = None


    if typed_question:

        question = typed_question.strip()


    elif st.session_state.pending_question:

        question = (
            st.session_state.pending_question
        )

        st.session_state.pending_question = None


    # ========================================================
    # PROCESS QUESTION
    # ========================================================

    if question:

        with st.chat_message("user"):

            st.markdown(question)


        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the insurance knowledge base..."
            ):

                try:

                    answer = generate_answer(
                        question,
                        st.session_state.provider,
                        st.session_state.api_key
                    )

                except Exception:

                    answer = (
                        "Unable to generate an answer. "
                        "Please check your API key and try again."
                    )


            st.markdown(answer)


        st.session_state.messages.append({

            "role": "user",

            "content": question

        })


        st.session_state.messages.append({

            "role": "assistant",

            "content": answer

        })


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">Star Health Insurance Knowledge Assistant</div>',
    unsafe_allow_html=True
)
