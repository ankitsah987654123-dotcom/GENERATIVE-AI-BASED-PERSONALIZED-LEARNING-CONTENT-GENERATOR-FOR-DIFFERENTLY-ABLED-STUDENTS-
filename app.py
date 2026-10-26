import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Accessible Learning Generator", page_icon="📘", layout="centered")

st.title("📘 Accessible Learning Content Generator")
st.caption("MTE prototype • Generative AI-based personalized learning for hearing-impaired students")

st.info(
    "Prototype status: initial UI and content-generation workflow. "
    "The local T5 model is optional and may need to download model files on first use."
)

with st.sidebar:
    st.header("Student preferences")
    level = st.selectbox("Explanation level", ["Simple English", "Very simple English", "Standard English"])
    output_type = st.selectbox("Content type", ["Short notes", "Simplified explanation", "Topic summary"])
    length = st.select_slider("Output length", options=["Short", "Medium", "Long"], value="Medium")
    use_t5 = st.checkbox("Try local T5 model (first run may download files)", value=False)

topic = st.text_input("Topic", placeholder="e.g., Photosynthesis, Operating System, Neural Networks")
source_text = st.text_area(
    "Paste learning material (optional)",
    height=180,
    placeholder="Paste a paragraph or lesson here. If left blank, enter a topic above."
)

def fallback_generate(topic_text, source, level, output_type, length):
    basis = source.strip() or f"Topic: {topic_text.strip()}"
    words = basis.replace("\n", " ").split()
    if length == "Short":
        limit = 35
    elif length == "Long":
        limit = 100
    else:
        limit = 60
    brief = " ".join(words[:limit])
    if len(words) > limit:
        brief += "..."
    if output_type == "Short notes":
        return (
            f"### Key Notes: {topic_text or 'Learning Topic'}\n"
            f"- Main idea: {brief}\n"
            "- Identify the important terms and their meanings.\n"
            "- Review the main steps or facts in the source material.\n"
            "- Practise by explaining the concept in your own words."
        )
    if output_type == "Topic summary":
        return (
            f"### Summary: {topic_text or 'Learning Topic'}\n"
            f"{brief}\n\n"
            "This is a basic preview generated from the text you provided. "
            "Check it against your original learning material."
        )
    return (
        f"### Simplified Explanation: {topic_text or 'Learning Topic'}\n"
        f"{brief}\n\n"
        "**Remember:** This preview shortens the supplied material; it does not yet perform full AI paraphrasing."
    )

def t5_generate(text, output_type, length):
    from transformers import pipeline
    generator = pipeline("text2text-generation", model="t5-small")
    task_prefix = "summarize: "
    max_len = {"Short": 60, "Medium": 100, "Long": 160}[length]
    result = generator(task_prefix + text, max_length=max_len, do_sample=False, truncation=True)
    return result[0]["generated_text"]

if st.button("Generate learning content", type="primary", use_container_width=True):
    if not topic.strip() and not source_text.strip():
        st.warning("Please enter a topic or paste learning material first.")
    else:
        input_text = source_text.strip() or f"Explain the topic {topic} in {level.lower()}."
        with st.spinner("Preparing learning content..."):
            if use_t5:
                try:
                    generated = t5_generate(input_text, output_type, length)
                    st.success("Generated using local T5-small model.")
                except Exception as exc:
                    generated = fallback_generate(topic, source_text, level, output_type, length)
                    st.warning("T5 could not be loaded. Showing a fallback preview instead. Check internet/model setup.")
                    st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
            else:
                generated = fallback_generate(topic, source_text, level, output_type, length)
                st.caption("Preview mode: the current output is rule-based, not a trained generative model.")
        st.markdown(generated)
        st.download_button(
            "Download content as TXT",
            data=generated,
            file_name="accessible_learning_content.txt",
            mime="text/plain",
            use_container_width=True
        )

st.divider()
st.subheader("Planned next steps")
st.markdown(
    "- Improve generation quality with a tested T5 model and task-specific prompts.\n"
    "- Add student profile and saved preferences.\n"
    "- Test readability and content accuracy with sample lessons.\n"
    "- Add accessibility improvements and evaluation results."
)