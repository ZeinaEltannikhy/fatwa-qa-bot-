import json
import os
import torch
from sentence_transformers import SentenceTransformer, util

# Load SentenceTransformer for semantic retrieval (embedding-based)
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Optionally save the model: embedding_model.save('./models/retriever')

# Load fatwa corpus once with error handling
fatwa_path = os.path.join("data", "processed", "fatwas_cleaned.jsonl")
if not os.path.exists(fatwa_path):
    raise FileNotFoundError(f"Fatwa data not found at {fatwa_path}")

with open(fatwa_path, 'r', encoding='utf-8') as f:
    fatwas = [json.loads(line) for line in f]
    corpus = [item["chunk"] for item in fatwas]
    meta = [(item["title"], item["url"]) for item in fatwas]

# Check if embeddings already exist to avoid recalculating
embeddings_path = os.path.join("data", "processed", "fatwa_embeddings.pt")

if os.path.exists(embeddings_path):
    corpus_embeddings = torch.load(embeddings_path)
else:
    corpus_embeddings = embedding_model.encode(corpus, convert_to_tensor=True)
    torch.save(corpus_embeddings, embeddings_path)

# 🧠 Retrieval using SentenceTransformer embeddings + cosine similarity
def retrieve_documents(question, top_k=3):
    # Get the embedding for the question
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    
    # Compute cosine similarity between the question and all corpus documents
    similarities = util.cos_sim(question_embedding, corpus_embeddings)[0]
    
    # Retrieve top_k most similar documents based on cosine similarity
    top_k_indices = similarities.topk(k=top_k).indices
    
    return [
        {"text": corpus[i], "title": meta[i][0], "url": meta[i][1]}
        for i in top_k_indices
    ]
