
# 🏥 Sağlık ve İlkyardım Chatbotu (RAG Mimarisi)

Bu proje, sağlık ve ilkyardım konularında kullanıcıya doğru bilgilendirme ve yönlendirme yapan bir **RAG (Retrieval-Augmented Generation)** tabanlı yapay zeka chatbotudur.

## 🎯 Projenin Amacı
Amaç, kullanıcıdan gelen doğal dildeki sağlık veya ilk yardım sorularını analiz ederek, en uygun yanıtı veri tabanından (bilgi tabanı) çekmek ve gerektiğinde LLM modeliyle desteklenmiş bir açıklama sunmaktır.

## 📚 Veri Seti
Veri seti üç dosyadan oluşur:
- `ilk_yardim_bilgileri.txt`: Temel ilk yardım bilgileri
- `saglik_onerileri.txt`: Günlük sağlık önerileri
- `acil_durumlar.txt`: Acil durum müdahale yönergeleri

## 🧠 Kullanılan Yöntemler
- **Retrieval**: Kullanıcının sorusuna benzer cümleleri bulmak için TF-IDF veya Embedding tabanlı arama.
- **Augmented Generation**: Bulunan bilgi parçası, büyük dil modeli (örneğin Gemini 2.5 Flash) tarafından genişletilerek anlamlı bir yanıt üretilir.

## ⚙️ Kurulum
1. Ortam oluşturun:
   ```bash
   python -m venv venv
   source venv/bin/activate  # (Windows: venv\Scripts\activate)
   ```
2. Gerekli kütüphaneleri kurun:
   ```bash
   pip install -r requirements.txt
   ```
3. Uygulamayı çalıştırın:
   ```bash
   streamlit run app.py
   ```

## ☁️ Web Arayüzü
Streamlit tabanlı bir sohbet arayüzü bulunmaktadır.
Kullanıcı “Elimi kestim, ne yapmalıyım?” gibi sorular yöneltebilir.

## 🌐 Deploy
Projeyi Streamlit Cloud veya HuggingFace Spaces üzerinde yayınlayabilirsiniz.

---
📍 **Hazırlayan:** Eda Çınar  
🔗 **Model:** Gemini 2.5 Flash (RAG Mimarisi)
