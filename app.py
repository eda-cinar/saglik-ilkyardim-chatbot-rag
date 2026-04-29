import streamlit as st
import os
import google.generativeai as genai  # Google'ın kendi kütüphanesi
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # Embedding için bu kalmalı

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="İlk Yardım Chatbotu-ACİLBOT", page_icon="💬")
st.title("🏥 Sağlık ve İlk Yardım Chatbotu-ACİLBOT")
st.write("RAG mimarili yapay zeka destekli sağlık asistanına hoş geldiniz!")

# --- API ANAHTARI KONTROLÜ (KRİTİK) ---
# Streamlit Cloud Secrets'tan anahtarı alıp sisteme tanıtıyoruz
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
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
        try:
            with open(file, "r", encoding="utf-8") as f:
                raw_text += f.read() + "\n"
        except FileNotFoundError:
            st.error(f"Hata: Veri dosyası bulunamadı: {file}")
            st.stop()

    # 2. Metin Parçalama
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=500, chunk_overlap=100, length_function=len
    )
    texts = text_splitter.split_text(raw_text)

    # 3. Embedding ve Vektör Veritabanı
    # google_api_key parametresini açıkça veriyoruz
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore


# Vektör veritabanını oluştur
try:
    vectorstore = setup_rag_environment()
except Exception as e:
    st.error("RAG ortamı başlatılamadı.")
    st.exception(e)
    st.stop()

# --- CHATBOT MANTIĞI ---
def rag_answer(query, vectorstore):
    # API Anahtarını yapılandır
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    
    # 1. Benzer dokümanları getir
    docs = vectorstore.similarity_search(query, k=4)
    context = "\n".join([doc.page_content for doc in docs])
    
    # 2. Prompt (İstem) oluştur
    prompt = f"""Sen bir ilk yardım asistanısın. Aşağıdaki bağlam bilgilerini kullanarak soruyu yanıtla.
    Eğer bilgi bağlamda yoksa 'Bu konuda bilgim bulunmamaktadır' de.

    BAĞLAM:
    {context}

    SORU: {query}
    YANIT:"""

    # 3. Modeli doğrudan çağır (404 hatası vermez)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    
    return response.text

# --- KULLANICI ARAYÜZÜ ---
user_input = st.text_input("Sorunuzu yazın (örn: Elimi kestim, ne yapmalıyım?):")

if st.button("Gönder"):
    if user_input.strip():
        with st.spinner('Yanıt oluşturuluyor...'):
            try:
                response = rag_answer(user_input, vectorstore)
                st.success(response)
            except Exception as e:
                st.error(f"Yanıt oluşturulurken bir hata oluştu: {e}")
    else:
        st.warning("Lütfen bir soru girin.")
