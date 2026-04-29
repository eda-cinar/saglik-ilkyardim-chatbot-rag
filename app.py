import streamlit as st
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

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
    # API Anahtarını yapılandır
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    # 1. Benzer dokümanları getir
    docs = _vectorstore.similarity_search(query, k=4)
    context = "\n".join([doc.page_content for doc in docs])
    
    # 2. KRİTİK DÜZELTME: v1 sürümünü açıkça zorluyoruz
    genai.configure(
        api_key=api_key,
        client_options={'api_version': 'v1'} # Hata burayı v1beta olarak gördüğü için oluşuyor
    )

    prompt = f"""Sen bir ilk yardım asistanısın. Aşağıdaki bağlamı kullanarak Türkçe yanıt ver.
    BAĞLAM: {context}
    SORU: {query}"""

    # 3. Modeli çağır
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    
    return response.text

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
