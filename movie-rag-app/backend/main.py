from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# === Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Models and Data ===
genai.configure(api_key="AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg")  # Replace securely in prod
model_genai = genai.GenerativeModel('gemini-1.5-flash')
model_embedding = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_pickle("quote_metadata.pkl")
embeddings = np.load("quote_embeddings.npy")
index = faiss.read_index("quote_index.faiss")

# === Pydantic Request Model ===
class QueryRequest(BaseModel):
    user_query: str

# === Gemini Wrapper ===
def gemini_response(prompt: str) -> str:
    try:
        response = model_genai.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("❌ Gemini generation error:", e)
        return "Sorry, I couldn't generate a quote at the moment."

# === Helper Function to Retrieve Quotes ===
def retrieve_similar_quotes(query, top_k=5):
    query_vec = model_embedding.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    results = []
    for idx in I[0]:
        row = df.iloc[idx]
        results.append({
            "text": row['text'],
            "character": row.get('character_name', 'Unknown'),
            "title": row.get('title', 'Unknown'),
            "year": row.get('year', 'Unknown'),
            "genres": row.get('genres', [])
        })
    return results

# === Quote Generation with Context ===
def generate_response_with_context(user_query, retrieved_quotes):
    context_lines = [
        f'"{q["text"]}" — {q["character"]} ({q["title"]}, {q["year"]}) [Genres: {q["genres"]}]'
        for q in retrieved_quotes
    ]

    context = "\n".join(context_lines)

    prompt = f"""You are a movie quote expert. The user asked: "{user_query}".
From the list of quotes below, choose the *most relevant* one that best responds to or reflects the user's query. Respond only with the *best matching quote*, without explanation.

Quotes:
{context}"""

    return gemini_response(prompt)


# === API Endpoint ===
@app.post("/ask")
async def ask_quotes(request: QueryRequest):
    retrieved = retrieve_similar_quotes(request.user_query)
    answer = generate_response_with_context(request.user_query, retrieved)
    return {
        "retrieved": retrieved,
        "answer": answer
    }