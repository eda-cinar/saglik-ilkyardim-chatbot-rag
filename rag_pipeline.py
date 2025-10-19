
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Veri yükleme
data_files = ["data/ilk_yardim_bilgileri.txt", "data/saglik_onerileri.txt", "data/acil_durumlar.txt"]
documents = []
for file in data_files:
    with open(file, "r", encoding="utf-8") as f:
        documents.extend(f.readlines())

# TF-IDF modelini oluştur
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

def retrieve(query, top_k=3):
    query_vec = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, doc_vectors).flatten()
    top_indices = sim_scores.argsort()[-top_k:][::-1]
    return [documents[i] for i in top_indices]

# rag_pipeline.py dosyasındaki rag_answer fonksiyonunu düzenleme

# rag_pipeline.py dosyasındaki rag_answer fonksiyonunun doğru hali

def rag_answer(query):
    retrieved_docs = retrieve(query)
    context = "\n".join(retrieved_docs)
    prompt = f"Aşağıdaki bilgiler ışığında soruya net bir yanıt ver:\n\nKontekst:\n{context}\n\nSoru: {query}\nYanıt:"
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # DÜZELTME: request_options={"timeout": 60} yerine 
    # timeout=60 parametresini doğrudan kullanın.
    response = model.generate_content(prompt, timeout=60) 
    
    return response.text
