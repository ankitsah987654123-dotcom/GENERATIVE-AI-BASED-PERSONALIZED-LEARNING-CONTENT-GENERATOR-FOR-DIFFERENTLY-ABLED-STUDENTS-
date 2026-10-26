# Generative AI-Based Personalized Learning Content Generator for Differently-Abled Students

A Streamlit web app that turns complex study material into short, simple English, designed to support **hearing-impaired students**, who often find long and complex sentences hard to read.

**Live app:** https://ujpjabjcabyx7ezhp6wxok.streamlit.app/
**Trained model:** https://huggingface.co/Ankit99716/simplifier-t5

---

## 1. Problem Statement
Many hearing-impaired students learn sign language first, so written English with long sentences and difficult vocabulary can be a barrier. This project generates simplified, easy-to-read learning content from any topic or lesson text.

## 2. Objectives
- Simplify complex academic text into short, plain sentences.
- Let students choose the explanation level, content type and output length.
- Work from either pasted text or just a topic name.
- Provide downloadable notes for offline revision.

## 3. Features
- **Topic mode:** enter a topic; the app fetches text from Wikipedia and simplifies it.
- **Paste mode:** paste your own lesson text and simplify it.
- **Content types:** Short notes, Simplified explanation, Topic summary.
- **Output length:** Short (2 sentences), Medium (5), Long (10).
- **Fallback mode:** a rule-based preview is used if the model cannot be loaded.
- **Download** the result as a TXT file.

## 4. Technologies Used
| Area | Tool |
|---|---|
| Language | Python |
| Web interface | Streamlit |
| Model | T5-small (fine-tuned) |
| Libraries | Transformers, PyTorch, SentencePiece, Datasets, Pandas |
| Model hosting | Hugging Face Hub |
| App deployment | Streamlit Community Cloud |

## 5. System Workflow
```
Topic / Pasted text
        |
        v
 (Topic only) Fetch text from Wikipedia
        |
        v
 Split text into sentences (2 / 5 / 10 by length setting)
        |
        v
 Fine-tuned T5 model simplifies each sentence ("simplify: ...")
        |
        v
 Format as notes / explanation / summary  ->  Display + Download
```

## 6. Dataset and Training
- `data.csv` contains sentence pairs: `source` (complex) and `target` (simple).
- `train.py` fine-tunes `t5-small` on this data (10 epochs, learning rate 3e-4) and saves the model to `model_dump/`.
- The model was uploaded to the Hugging Face Hub and is loaded by `app.py`.

## 7. Project Structure
```
app.py             Streamlit application
train.py           Model fine-tuning script
data.csv           Training dataset (complex -> simple pairs)
requirements.txt   Dependencies for the deployed app
```

## 8. How to Run Locally
```bash
git clone https://github.com/ankitsah987654123-dotcom/GENERATIVE-AI-BASED-PERSONALIZED-LEARNING-CONTENT-GENERATOR-FOR-DIFFERENTLY-ABLED-STUDENTS-.git
cd GENERATIVE-AI-BASED-PERSONALIZED-LEARNING-CONTENT-GENERATOR-FOR-DIFFERENTLY-ABLED-STUDENTS-
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model: `pip install datasets accelerate pandas`, edit `data.csv`, then run `python train.py`.

## 9. Limitations
- The current dataset is small (about 30 pairs), so simplification quality is limited.
- The explanation level setting does not yet change the model's output style.
- The Wikipedia source is English only and its content is not verified for accuracy.
- Output has not yet been formally evaluated with readability or accuracy metrics.

## 10. Future Work
- Train on a larger simplification dataset (for example ASSET or WikiAuto).
- Add level-specific training so "Very simple English" differs from "Simple English".
- Evaluate with Flesch Reading Ease, SARI and ROUGE scores.
- Add text-to-sign-language support, text-to-speech and student profiles.
- Test with hearing-impaired students and teachers.

## 11. Author
Ankit Sah
