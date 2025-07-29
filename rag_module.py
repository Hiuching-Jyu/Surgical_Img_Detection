# rag_module.py
import os
import faiss
import pytesseract
import pdfplumber
from PIL import Image
from typing import List
# from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import openai

# from llama_index.embeddings import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer
import tempfile

KNOWLEDGE_DIR = "RAG_Knowledge"
EMBED_MODEL = "BAAI/bge-small-en"
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

def extract_text_from_docs(folder):
    texts = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if fname.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        elif fname.endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                texts.append("\n".join(page.extract_text() or "" for page in pdf.pages))
        elif fname.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                img = Image.open(path)
                texts.append(pytesseract.image_to_string(img))
            except:
                print(f"⚠️ Failed to OCR {path}")
    return texts

def build_vector_index():
    print("📥 Loading documents from:", KNOWLEDGE_DIR)
    docs = extract_text_from_docs(KNOWLEDGE_DIR)
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, txt in enumerate(docs):
            with open(os.path.join(temp_dir, f"doc_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(txt)
        reader = SimpleDirectoryReader(temp_dir)
        docs_nodes = reader.load_data()
    print("✅ Documents loaded into index.")
    index = VectorStoreIndex.from_documents(docs_nodes, embed_model=embed_model)
    return index

# 构建一次，后续可优化为持久化载入
index = build_vector_index()
query_engine = index.as_query_engine(similarity_top_k=3)

def retrieve_step_by_rag(description_text: str) -> str:
    query = (
        "You are a surgical assistant system. Based on the following image description, "
        "determine which surgical step is currently taking place. "
        "Choose one from: Preparation & Exposure, Dissection & Vessel Control, Uterus Removal & Closure.\n\n"
        f"Image description: {description_text.strip()}"
    )
    response = query_engine.query(query)
    return str(response).strip()

