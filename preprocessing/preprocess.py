import json
import os
import re
from tqdm import tqdm

INPUT_PATH = "data/fatwas.json"
OUTPUT_PATH = "data/processed/fatwas_cleaned.jsonl"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_words=100):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_file:
    for entry in tqdm(data):
        title = clean_text(entry.get("title", ""))
        question = clean_text(entry.get("question", ""))
        answer = clean_text(entry.get("answer", ""))
        chunks = chunk_text(answer)

        for chunk in chunks:
            json.dump({
                "title": title,
                "url": entry["url"],
                "question": question,
                "chunk": chunk
            }, out_file, ensure_ascii=False)
            out_file.write('\n')

print(f"✅ Saved cleaned data to {OUTPUT_PATH}")

# preprocessing/preprocess.py

# preprocessing/preprocess.py
