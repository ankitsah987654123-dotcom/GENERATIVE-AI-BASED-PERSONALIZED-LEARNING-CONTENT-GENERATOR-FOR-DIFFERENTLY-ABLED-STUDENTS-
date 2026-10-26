import re
import json
import urllib.parse
import urllib.request
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Accessible Learning Generator", page_icon="📘", layout="centered")

st.title("📘 Accessible Learning Content Generator")
st.caption("Generative AI-based personalized learning for hearing-impaired students")

PREFIX = "simplify: "                      # must match train.py
LOCAL_MODEL_DIR = Path("model_dump")       # folder created by train.py
HF_MODEL_ID = "Ankit99716/simplifier-t5"                           # e.g. "yourname/simplifier-t5" (Hugging Face Hub)

model_available = LOCAL_MODEL_DIR.exists() or bool(HF_MODEL_ID)
if model_available:
    st.info("Fine-tuned simplification model is available.")
else:
    st.info("Trained model not found yet. The app will use the rule-based fallback.")

with st.sidebar:
    st.header("Student preferences")
    level = st.selectbox("Explanation level", ["Simple English", "Very simple English", "Standard English"])
    output_type = st.selectbox("Content type", ["Short notes", "Simplified explanation", "Topic summary"])
    length = st.select_slider("Output length", options=["Short", "Medium", "Long"], value="Medium")
    use_model = st.checkbox("Use trained model", value=model_available)

topic = st.text_input("Topic", placeholder="e.g., Photosynthesis, Operating System, Neural Networks")
source_text = st.text_area(
    "Paste learning material",
    height=180,
    placeholder="Paste a paragraph or lesson here. The trained model simplifies this text."
)


@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    path = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else HF_MODEL_ID
    if not path:
        raise FileNotFoundError("No model_dump folder and no HF_MODEL_ID set.")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    model.eval()
    return tok, model


def model_generate(text, output_type, length):
    tok, model = load_model()
    max_new = {"Short": 48, "Medium": 80, "Long": 128}[length]
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = [" ".join(sents[i:i + 2]) for i in range(0, len(sents), 2)]
    outputs = []
    for chunk in chunks:
        enc = tok(PREFIX + chunk, return_tensors="pt", truncation=True, max_length=256)
        ids = model.generate(**enc, max_new_tokens=max_new, num_beams=4,
                             no_repeat_ngram_size=3, early_stopping=True)
        outputs.append(tok.decode(ids[0], skip_special_tokens=True))
    result = " ".join(outputs)
    if output_type == "Short notes":
        return "### Key Notes\n" + "\n".join(
            f"- {s}" for s in re.split(r"(?<=[.!?])\s+", result) if s)
    return result


@st.cache_data(show_spinner=False, ttl=3600)
def get_wikipedia_text(topic_text):
    """Fetch a short plain-text summary of the topic from Wikipedia."""
    headers = {"User-Agent": "AccessibleLearningApp/1.0"}
    def fetch(url):
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    try:
        q = urllib.parse.quote(topic_text.strip())
        found = fetch(f"https://en.wikipedia.org/w/api.php?action=opensearch&search={q}&limit=1&format=json")
        title = found[1][0] if found[1] else topic_text.strip()
        data = fetch("https://en.wikipedia.org/api/rest_v1/page/summary/" +
                     urllib.parse.quote(title.replace(" ", "_")))
        return data.get("extract", "")
    except Exception:
        return ""


def fallback_generate(topic_text, source, level, output_type, length):
    basis = source.strip() or f"Topic: {topic_text.strip()}"
    words = basis.replace("\n", " ").split()
    limit = {"Short": 35, "Medium": 60, "Long": 100}[length]
    brief = " ".join(words[:limit]) + ("..." if len(words) > limit else "")
    if output_type == "Short notes":
        return (f"### Key Notes: {topic_text or 'Learning Topic'}\n- Main idea: {brief}\n"
                "- Identify the important terms and their meanings.\n"
                "- Practise by explaining the concept in your own words.")
    if output_type == "Topic summary":
        return f"### Summary: {topic_text or 'Learning Topic'}\n{brief}"
    return f"### Simplified Explanation: {topic_text or 'Learning Topic'}\n{brief}"


if st.button("Generate learning content", type="primary", use_container_width=True):
    if not topic.strip() and not source_text.strip():
        st.warning("Please enter a topic or paste learning material first.")
    else:
        input_text = source_text.strip()
        if not input_text and topic.strip():
            with st.spinner('Fetching topic information...'):
                input_text = get_wikipedia_text(topic)
            if input_text:
                st.caption('Source text fetched from Wikipedia and simplified by the model.')
            else:
                st.warning('Could not find information on this topic. Please paste some text.')
        with st.spinner("Preparing learning content..."):
            if use_model and input_text:
                try:
                    generated = model_generate(input_text, output_type, length)
                    st.success("Generated using the fine-tuned model.")
                except Exception as exc:
                    generated = fallback_generate(topic, source_text, level, output_type, length)
                    st.warning("Trained model could not be loaded. Showing fallback preview.")
                    st.caption(f"{type(exc).__name__}: {exc}")
            else:
                if use_model and not input_text:
                    st.warning("The trained model simplifies pasted text. Paste material for best results.")
                generated = fallback_generate(topic, input_text or source_text, level, output_type, length)
                st.caption("Preview mode: rule-based output.")
        st.markdown(generated)
        st.download_button("Download content as TXT", data=generated,
                           file_name="accessible_learning_content.txt",
                           mime="text/plain", use_container_width=True)
