
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
I. Retrieval (Bilgi Çekme) Aşamasında Kullanılan Yöntemler (3 Teknik)
Bu aşamada amaç, kullanıcının sorusuyla en alakalı metin parçalarını (konteksti) veri setinizden bulmaktır.
1.	Metin Parçalama (Chunking):
o	Yöntem: CharacterTextSplitter kullanarak büyük metin belgelerinizi (.txt dosyaları) yönetilebilir, küçük parçalara (chunk) bölmek.
o	Amaç: Anlamı kaybetmeden, her bir parçanın Embeddings modeli tarafından etkin bir şekilde vektöre dönüştürülmesini sağlamak ve Büyük Dil Modeli'ne (LLM) gereksiz bilgi göndermekten kaçınmaktır.
2.	Embedding (Vektörleştirme):
o	Yöntem: GoogleGenerativeAIEmbeddings ve models/text-embedding-004 modelini kullanarak metin parçalarını yüksek boyutlu sayısal vektörlere dönüştürmek.
o	Amaç: İnsan dilindeki anlamsal anlamı yakalamak. Yani "elimi kestim" ile "kesik tedavisi" kelimeleri farklı olsa da, vektör uzayında birbirine yakın olacaklardır.
3.	Vektör Benzerliği Araması ve Depolama:
o	Yöntem: FAISS (Facebook AI Similarity Search) vektör veritabanını kullanarak, kullanıcının sorusunun vektörü ile depolanan tüm metin parçalarının vektörleri arasındaki Kosinüs Benzerliğini hesaplamak.
o	Amaç: Veri setinizdeki en alakalı (en yüksek benzerlik skoruna sahip) $K$ adet belgeyi (bizim projemizde $K=4$) hızla çekmektir.
________________________________________
II. Augmented Generation (Genişletilmiş Yanıt Üretme) Aşamasında Kullanılan Yöntemler (2 Teknik)
Bu aşamada amaç, çekilen bilgiyi kullanarak kullanıcıya akıcı ve doğru bir yanıt oluşturmaktır.
4.	Prompt Template (İstem Şablonu Kullanımı):
o	Yöntem: LangChain'den alınan PromptTemplate ile LLM'ye ne yapması gerektiğini açıkça belirten yapılandırılmış bir talimat (istem) göndermek.
o	Amaç: LLM'nin rolünü belirlemek ("ilk yardım ve sağlık asistanı"), çekilen bilgiyi bağlam ({context}) içine yerleştirmek ve LLM'nin yanıt formatını kontrol etmek ("Verilen bilgilerde bu konu hakkında bilgi bulunmamaktadır." gibi bir kısıtlama koymak).
5.	LLM ile Yanıt Üretme (Generation):
o	Yöntem: Gemini 2.5 Flash (hız ve maliyet etkinliği için) büyük dil modelini kullanarak, çekilen bağlam bilgileri ve kullanıcı sorusu ışığında son, doğal dildeki yanıtı oluşturmak.
o	Amaç: Çekilen ham bilgiyi alarak, bağlam içinde mantıklı, akıcı ve Türkçe bir cevaba dönüştürmektir.
Bu beş yöntem, RAG mimarisinin temel taşlarını oluşturur ve chatbotumun başarılı bir şekilde çalışmasını sağlar.


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
<img width="976" height="346" alt="Ekran görüntüsü 2025-10-21 210505" src="https://github.com/user-attachments/assets/dff7b29a-a93f-454a-ac84-745fa4bbfaed" />

<img width="962" height="402" alt="Ekran görüntüsü 1 2025-10-21 210615" src="https://github.com/user-attachments/assets/356ca2d6-5ce0-458d-a51e-6a75d5c09afe" />

<img width="1022" height="503" alt="Ekran görüntüsü 2 2025-10-21 210645" src="https://github.com/user-attachments/assets/c3f1e07e-00e1-4d3c-a815-734bc5b00fa6" />



chatbot web: https://saglik-ilkyardim-chatbot-rag-hgcostnve4dah9w7xa7adu.streamlit.app/



Contact
Email:edabattalcinar@hotmail.com
GitHub:https://github.com/eda-cinar
Linkedin:https://www.linkedin.com/in/eda-çinar-2a8a2b58/



---
📍 **Hazırlayan:** Eda Çınar  
🔗 **Model:** Gemini 2.5 Flash (RAG Mimarisi)
