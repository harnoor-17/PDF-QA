import os
import faiss
import PyPDF2
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini = genai.GenerativeModel("gemini-1.5-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chunks = []
index = None

def split_text(text, max_tokens=200):
    words = text.split()
    return [' '.join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

def process_pdf(file_path):
    global chunks, index
    chunks.clear()
    reader = PyPDF2.PdfReader(file_path)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            chunks.extend(split_text(text))
    embeddings = embedder.encode(chunks).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

def get_top_chunks(query, k=3):
    query_vector = embedder.encode([query]).astype("float32")
    _, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]

def answer_question(query):
    if not index:
        return "⚠️ Please upload a PDF first."
    context = "\n".join(get_top_chunks(query))
    prompt = f"Answer based only on this context:\n\n{context}\n\nQuestion: {query}"
    try:
        response = gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error: {e}"
