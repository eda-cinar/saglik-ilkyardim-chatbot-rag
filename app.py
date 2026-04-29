import streamlit as st
import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="İlk Yardım Chatbotu-ACİLBOT", page_icon="💬")
st.title("🏥 Sağlık ve İlk Yardım Chatbotu-ACİLBOT")

# --- API ANAHTARI KONTROLÜ ---
# Streamlit Secrets'ta anahtar isminin GROQ_API_KEY olduğundan emin ol
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    st.error("Lütfen Secrets kısmına 'GROQ_API_KEY' ekleyin!")
    st.stop()

# --- RAG ORTAMI ---
@st.cache_resource
def setup_rag_environment():
    data_files = ["data/ilk_yardim_bilgileri.txt", "data/saglik_onerileri.txt", "data/acil_durumlar.txt"]
    raw_text = ""
    for file in data_files:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                raw_text += f.read() + "\n"
    
    if not raw_text:
        st.error("Veri dosyaları bulunamadı!")
        st.stop()

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(raw_text)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

vectorstore = setup_rag_environment()

# --- CHATBOT MANTIĞI (GROQ KULLANIMI) ---
def rag_answer(query, _vectorstore):
    client = Groq(api_key=GROQ_API_KEY)
    
    # RAG: Veri çekme
    docs = _vectorstore.similarity_search(query, k=4)
    context = "\n".join([doc.page_content for doc in docs])
    
    # --- SINIRLANDIRILMIŞ SİSTEM TALİMATI ---
    system_prompt = """
    Sen bir ACİLBOT isimli İlk Yardım ve Sağlık asistanısın. 
    Görevin SADECE ilk yardım, sağlık önerileri ve acil durum müdahaleleri hakkında bilgi vermektir.
    
    KURALLAR:
    1. Eğer soru ilk yardım, sağlık veya tıbbi konularla ilgili DEĞİLSE (örneğin: yemek tarifi, matematik, futbol, genel kültür vb.), şu cevabı ver: 
       'Ben sadece ilk yardım ve sağlık konularında bilgi vermek üzere eğitildim. Lütfen bu alanlarla ilgili bir soru sorun.'
    2. Yanıtlarını SADECE sana verilen BAĞLAM bilgilerine dayandır.
    3. Bağlamda bilgi yoksa, bilmediğini nazikçe belirt.
    4. Acil durumlarda mutlaka '112 Acil Servis'i arayın' uyarısını yap.
    """

    user_message = f"BAĞLAM:\n{context}\n\nSORU: {query}"

    # Groq Llama 3 modelini çağır
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt}, # Kimlik burada tanımlanıyor
            {"role": "user", "content": user_message}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.1, # Daha ciddi ve tutarlı cevaplar için sıcaklığı düşürdük
    )
    return chat_completion.choices[0].message.content
# --- ARAYÜZ ---
user_input = st.text_input("Sorunuzu yazın:")

if st.button("Gönder"):
    if user_input:
        with st.spinner('Yanıt oluşturuluyor...'):
            try:
                answer = rag_answer(user_input, vectorstore)
                st.success(answer)
            except Exception as e:
                st.error(f"Hata oluştu: {e}")
