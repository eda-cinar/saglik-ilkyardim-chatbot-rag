import streamlit as st
from rag_pipeline import rag_answer

st.set_page_config(page_title="İlk Yardım Chatbotu", page_icon="💬")

st.title("🏥 Sağlık ve İlk Yardım Chatbotu")
st.write("RAG mimarili yapay zeka destekli sağlık asistanına hoş geldiniz!")

user_input = st.text_input("Sorunuzu yazın (örn: Elimi kestim, ne yapmalıyım?):")

if st.button("Gönder"):
    if user_input.strip():
        response = rag_answer(user_input)
        st.success(response)
    else:
        st.warning("Lütfen bir soru girin.")
