import os
from dotenv import load_dotenv
import streamlit as st
from datasets import load_dataset
from google import genai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------
# Ortam değişkenlerini yükle
# -------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Gemini API client
client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------
# Streamlit başlık
# -------------------
st.set_page_config(page_title="📚 Kitapyurdu RAG Chatbot", page_icon="📖")
st.title("📚 Kitapyurdu RAG Chatbot")

# -------------------
# Dataset yükleme
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
# Metinleri böl ve embedding oluştur
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
# Chat fonksiyonu
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
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        temperature=0.2
    )
    return response.text

# -------------------
# Streamlit arayüz
# -------------------
user_input = st.text_input("Sorunuzu yazın:")
if st.button("Gönder"):
    if user_input:
        with st.spinner("Cevap oluşturuluyor..."):
            answer = generate_answer(user_input)
        st.write("💬 Cevap:", answer)
