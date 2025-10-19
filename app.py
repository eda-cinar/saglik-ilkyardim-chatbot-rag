import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# API Anahtarı Ayarı (Sadece güvenilir bir Streamlit Secrets kullanımı için)
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # Artık LLM/Embeddings sınıfına direkt verilecek

st.set_page_config(page_title="İlk Yardım Chatbotu", page_icon="💬")
st.title("🏥 Sağlık ve İlk Yardım Chatbotu")
st.write("RAG mimarili yapay zeka destekli sağlık asistanına hoş geldiniz!")

# --- FAISS VE EMBEDDINGS OLUŞTURMA (SADECE BİR KEZ ÇALIŞIR) ---
@st.cache_resource
def setup_rag_environment():
    # 1. Veri Yükleme ve Parçalama
    data_files = ["data/ilk_yardim_bilgileri.txt", "data/saglik_onerileri.txt", "data/acil_durumlar.txt"]
    raw_text = ""
    for file in data_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                raw_text += f.read() + "\n"
        except FileNotFoundError:
            # Gerekirse bu uyarıyı kullanıcıya gösterebiliriz
            pass

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=500, chunk_overlap=100, length_function=len
    )
    texts = text_splitter.split_text(raw_text)

    # 2. Embedding Modeli ve Vektör Veritabanı Oluşturma
    # API key, os.getenv'den otomatik olarak okunur.
    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001") 
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    return vectorstore

# Vektör veritabanını oluştur veya cache'den yükle
try:
    vectorstore = setup_rag_environment()
except Exception as e:
    st.error("RAG ortamı başlatılamadı. API Anahtarınızı ve Billing ayarlarınızı kontrol edin.")
    st.exception(e)
    st.stop()


# --- CHATBOT MANTIĞI (rag_answer fonksiyonu app.py'ye taşındı) ---
def rag_answer(query, vectorstore):
    # LLM Modelini Tanımlama
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
        # API key ortam değişkeninden alınacak
    )
    
    # 1. Retrieval (Arama) Bileşenini Tanımlama
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 2. Prompt Template (İstemi) Oluşturma
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
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    response = qa_chain.run(query)
    
    return response


# --- KULLANICI ARAYÜZÜ ---
user_input = st.text_input("Sorunuzu yazın (örn: Elimi kestim, ne yapmalıyım?):")

if st.button("Gönder"):
    if user_input.strip():
        # rag_answer şimdi vectorstore'u alıyor
        with st.spinner('Yanıt oluşturuluyor...'):
            response = rag_answer(user_input, vectorstore)
        st.success(response)
    else:
        st.warning("Lütfen bir soru girin.")
