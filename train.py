"""
Fine-tune t5-small for text simplification (run once, e.g. on Google Colab).
Input : data.csv  (columns: source,target)
Output: model_dump/  (this folder is your "dump" - the trained model)
"""
import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

PREFIX = "simplify: "          # must match PREFIX in app.py
BASE = "t5-small"
OUT = "model_dump"

df = pd.read_csv("data.csv").dropna()
ds = Dataset.from_pandas(df[["source", "target"]])

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE)

def prep(batch):
    enc = tok([PREFIX + s for s in batch["source"]], max_length=256, truncation=True)
    enc["labels"] = tok(text_target=batch["target"], max_length=128, truncation=True)["input_ids"]
    return enc

ds = ds.map(prep, batched=True, remove_columns=ds.column_names)

args = Seq2SeqTrainingArguments(
    output_dir="checkpoints",
    num_train_epochs=10,            # tiny data -> more epochs; use 3-5 for 10k+ rows
    per_device_train_batch_size=8,
    learning_rate=3e-4,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
)
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds,
                         data_collator=DataCollatorForSeq2Seq(tok, model=model))
trainer.train()

trainer.save_model(OUT)
tok.save_pretrained(OUT)
print("Saved to", OUT)
