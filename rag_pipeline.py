
import os
import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer # KALDIRILACAK
# from sklearn.metrics.pairwise import cosine_similarity # KALDIRILACAK
import google.generativeai as genai

# YENİ EKLEMELER
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



# Veri yükleme
data_files = ["data/ilk_yardim_bilgileri.txt", "data/saglik_onerileri.txt", "data/acil_durumlar.txt"]

# Tüm verileri tek bir metin bloğunda birleştirme
raw_text = ""
for file in data_files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            raw_text += f.read() + "\n"
    except FileNotFoundError:
        print(f"Uyarı: {file} bulunamadı.")


# 1. Metni Parçalama (Chunking)
# LangChain kullanarak metni daha küçük, anlamlı parçalara ayırma
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# 2. Embedding Modeli ve Vektör Veritabanı Oluşturma
# Gemini'nin embedding modelini kullanma
embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")

# FAISS (Vektör Veritabanı) oluşturma
# Bu adım, her bir metin parçasını vektöre çevirip veritabanına kaydeder
vectorstore = FAISS.from_texts(texts, embeddings)


# Eski TF-IDF retrieval fonksiyonu kaldırılır.
# def retrieve(query, top_k=3): ...

# rag_pipeline.py dosyasındaki rag_answer fonksiyonunu düzenleme



def rag_answer(query):
    # LLM Modelini Tanımlama
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        # API anahtarı genai.configure() ile otomatik alınır
    )
    
    # 1. Retrieval (Arama) Bileşenini Tanımlama
    # Vektör veritabanını arama aracı olarak kullanır (k=3 ile en alakalı 3 belgeyi çeker)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 2. Prompt Template (İstemi) Oluşturma
    # LLM'ye ne yapması gerektiğini söyleyen yönerge
    prompt_template = """Aşağıdaki bağlam bilgileri, ilk yardım ve sağlık konularında hazırlanmıştır. 
    Verilen bağlamı kullanarak, kullanıcı sorusuna net ve güvenilir bir Türkçe yanıt ver. 
    Bağlamda bulunmayan bir bilgi sorulursa, "Verilen bilgilerde bu konu hakkında bilgi bulunmamaktadır." diye yanıtla.

    BAĞLAM:
    {context}

    SORU: {question}
    YANIT:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    # 3. RAG Zincirini Oluşturma (Retrieval + Generation)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Bütün bağlamı tek bir prompt'a sıkıştırır
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    # Zinciri Çalıştırma
    response = qa_chain.run(query)
    
    return response
