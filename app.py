import os
from dotenv import load_dotenv
import streamlit as st
from datasets import load_dataset
import google.generativeai as genai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------
# 0. Ortam değişkenlerini yükle
# -------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

genai.configure(api_key=GEMINI_API_KEY)

# -------------------
# 1. Streamlit başlık
# -------------------
st.set_page_config(page_title="📚 Kitapyurdu RAG Chatbot", page_icon="📖")
st.title("📚 Kitapyurdu RAG Chatbot")

# -------------------
# 2. Dataset yükleme
# -------------------
@st.cache_data
def load_kitapyurdu_dataset():
    dataset = load_dataset(
        "alibayram/kitapyurdu_yorumlar",
        use_auth_token=HF_TOKEN
    )
    return dataset['train']

dataset = load_kitapyurdu_dataset()
st.write(f"✅ Dataset yüklendi. Toplam yorum: {len(dataset)}")

# -------------------
# 3. Metinleri böl ve embedding oluştur
# -------------------
@st.cache_resource
def create_vectorstore(dataset):
    texts = [row['yorum'] for row in dataset]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_texts(texts)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(docs, embeddings, persist_directory="./chroma_db")
    vectordb.persist()
    return vectordb

vectordb = create_vectorstore(dataset)
st.write("✅ Vectorstore hazır.")

# -------------------
# 4. Chat fonksiyonu
# -------------------
def generate_answer(query):
    # Benzer metinleri retrieval
    docs = vectordb.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])
    
    # Profesyonel prompt
    prompt = f"""
Aşağıdaki kullanıcı yorumlarını göz önünde bulundurarak soruyu yanıtla.
Yorumlar:
{context}

Kullanıcı sorusu: {query}

Yanıtını kısa, anlaşılır ve bilgilendirici şekilde ver.
"""
    # Gemini ile cevap oluştur
    response = genai.chat.create(
        model="gemini-2.0-flash",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    return response.last

# -------------------
# 5. Streamlit arayüz
# -------------------
user_input = st.text_input("Sorunuzu yazın:")
if st.button("Gönder"):
    if user_input:
        with st.spinner("Cevap oluşturuluyor..."):
            answer = generate_answer(user_input)
        st.write("💬 Cevap:", answer)
