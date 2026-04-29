import streamlit as st
import os
import requests
import json
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="İlk Yardım Chatbotu-ACİLBOT", page_icon="💬")
st.title("🏥 Sağlık ve İlk Yardım Chatbotu-ACİLBOT")
st.write("RAG mimarili yapay zeka destekli sağlık asistanına hoş geldiniz!")

# --- API ANAHTARI KONTROLÜ ---
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
else:
    st.error("Lütfen Streamlit Cloud Secrets kısmına 'GOOGLE_API_KEY' ekleyin!")
    st.stop()

# --- FAISS VE EMBEDDINGS OLUŞTURMA ---
@st.cache_resource
def setup_rag_environment():
    # 1. Veri Yükleme
    data_files = ["data/ilk_yardim_bilgileri.txt", "data/saglik_onerileri.txt", "data/acil_durumlar.txt"]
    raw_text = ""
    for file in data_files:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                raw_text += f.read() + "\n"
        else:
            st.warning(f"Uyarı: {file} dosyası bulunamadı, bu dosya atlanıyor.")
    
    if not raw_text:
        st.error("Hiçbir veri dosyası yüklenemedi! Lütfen data/ klasörünü kontrol edin.")
        st.stop()

    # 2. Metin Parçalama
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=500, chunk_overlap=100, length_function=len
    )
    texts = text_splitter.split_text(raw_text)

    # 3. Embedding (HuggingFace kullanarak Google API hatasını bypass ediyoruz)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

# Vektör veritabanını oluştur
try:
    vectorstore = setup_rag_environment()
except Exception as e:
    st.error(f"RAG ortamı başlatılamadı: {e}")
    st.stop()

# --- CHATBOT MANTIĞI ---
def rag_answer(query, _vectorstore):
    # Groq API anahtarını Secrets'tan al (İsmini GROQ_API_KEY yapabilirsin)
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    # RAG: Veri çekme kısmı aynı kalıyor
    docs = _vectorstore.similarity_search(query, k=4)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Prompt
    prompt = f"Bağlam: {context}\n\nSoru: {query}\n\nYanıtı Türkçe ver:"

    # Groq üzerinden Llama 3 modelini çağırıyoruz (Işık hızında çalışır)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    
    return chat_completion.choices[0].message.content

    # 3. İstek Atma
    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = response.json()

    if response.status_code == 200:
        try:
            return result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return "Yanıt formatı çözülemedi."
    else:
        return f"Hata: {response.status_code}. Mesaj: {result.get('error', {}).get('message', 'Bilinmeyen Hata')}"

    # 4. Doğrudan İstek Atma
    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = response.json()

    if response.status_code == 200:
        try:
            return result['candidates'][0]['content']['parts'][0]['text']
        except:
            return "Yanıt işlenirken bir hata oluştu."
    else:
        # Eğer hala 404 veriyorsa bu sefer hatanın detayını göreceğiz
        return f"Hata Kodu: {response.status_code} - Mesaj: {result.get('error', {}).get('message', 'Bilinmeyen Hata')}"

# --- KULLANICI ARAYÜZÜ ---
user_input = st.text_input("Sorunuzu yazın (örn: Elimi kestim, ne yapmalıyım?):")

if st.button("Gönder"):
    if user_input.strip():
        with st.spinner('Yanıt oluşturuluyor...'):
            try:
                # Vektör veritabanını fonksiyona gönderiyoruz
                answer = rag_answer(user_input, vectorstore)
                st.success(answer)
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
                st.info("İpucu: Eğer 404 hatası alıyorsanız, Google API anahtarınızın bu modeli desteklediğinden emin olun veya model ismini 'gemini-pro' olarak güncelleyin.")
    else:
        st.warning("Lütfen bir soru girin.")
