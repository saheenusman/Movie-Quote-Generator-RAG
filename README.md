
# 🎬 Movie Quote Generator using RAG + FAISS

A simple Retrieval-Augmented Generation (RAG) system that generates movie quotes based on user queries. The system uses **FAISS** for fast vector search, **Gemini 1.5 Flash** for LLM responses, and a lightweight **HTML + TailwindCSS** frontend served via FastAPI.

---

## 🗂️ Dataset

- **Cornell Movie Dialogs Corpus**  
  [https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

---

## 🛠️ Project Structure

```
.
├── clean.ipynb                 # Data cleaning
├── data_prep.ipynb             # Preparing and formatting the dataset
├── embeding.ipynb              # Generating embeddings and building FAISS index
├── simple rag output.ipynb     # Testing RAG pipeline
├── main.py                     # FastAPI backend server
├── templates/index.html        # Frontend UI (TailwindCSS + JS)
├── requirements.txt            # Python dependencies
└── README.md                   # Project description
```

---

## ⚙️ How It Works

### 1️⃣ Data Preparation & Embedding

- Raw movie dialogs from the Cornell corpus are cleaned and parsed into individual quotes.
- Gemini embedding model is used to generate vector representations for each quote.
- The resulting vectors are stored in a **FAISS index** (`.faiss`, `.pkl`, `.npy` files).

### 2️⃣ RAG Pipeline at Runtime

- User submits a query (e.g. "love", "friendship", "betrayal") through the web UI.
- Backend uses FAISS to retrieve top-N similar movie quotes based on vector similarity.
- Retrieved quotes are passed as context to Gemini 1.5 Flash LLM.
- LLM generates a final response — which may include reworded, combined, or directly retrieved quotes — and returns it to frontend.

### 3️⃣ Frontend

- Built with **HTML + TailwindCSS** for a simple, clean UI.
- Calls FastAPI `/ask` endpoint with user query.
- Displays matching movie quotes dynamically on the page.

---

## 📦 FAISS Index Files

Since GitHub does not allow uploading large files directly in the repository,  
the **FAISS index** files (`.pkl`, `.npy`, `.faiss`) are available under:

👉 **GitHub Releases → Assets**

Please download the latest release and place the files in the appropriate directory as used by `main.py`.

---

## 🚀 Running the Project Locally

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Start the FastAPI backend

```bash
python -m uvicorn main:app --reload
```

### 3️⃣ Open the Frontend

Simply open this file in your browser:

```text
frontend/index.html
```

The page will send POST requests to:

```text
http://127.0.0.1:8000/ask
```

---

## 🧠 Technologies Used

- FAISS (Vector Search)
- Gemini 1.5 Flash (LLM + embeddings)
- FastAPI (API backend)
- TailwindCSS (Frontend UI)
- Cornell Movie Dialogs Corpus (Dataset)

---

## ⭐ Credits

- Dataset: [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
- Developed by: Saheen Usman
