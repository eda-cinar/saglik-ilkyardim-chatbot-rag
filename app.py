import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings

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
    # Model ismini güncelledik: gemini-1.5-flash
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", # Ön eki kaldırarak sadece ismi yaz
    temperature=0.3,
    google_api_key=os.environ["GOOGLE_API_KEY"]
 )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
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
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain.run(query)

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
