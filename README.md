# 🏥 MediBot: AI-Powered Medical Assistant

MediBot ek advanced **Generative AI** chatbot hai jo medical documents (PDFs) se context read karke user ke sawalon ka sahi aur concise jawab deta hai. Yeh **RAG (Retrieval-Augmented Generation)** pipeline ka upyog karta hai.

---

## 🚀 Features
* **RAG Integration**: Medical PDFs se relevant data retrieve karke answers generate karta hai.
* **Fast Inference**: Groq Cloud ka use karke **Llama-3.1** models ke saath super-fast response deta hai.
* **Medical Theme UI**: Streamlit par banaya gaya ek clean, dark-themed medical interface.
* **Glassmorphism Design**: Modern UI jisme black text aur professional spacing ka dhyan rakha gaya hai.
* **Local & Cloud Support**: Ise local machine aur Streamlit Community Cloud dono par deploy kiya ja sakta hai.

---

## 🛠️ Tech Stack
* **Language**: Python
* **Framework**: LangChain
* **Frontend**: Streamlit
* **LLM**: Groq (Llama-3.1-8b-instant)
* **Embeddings**: Hugging Face (all-MiniLM-L6-v2)
* **Vector Database**: FAISS

---

## 📂 Project Structure
```text
├── medibot.py               # Main Streamlit Application
├── ingest.py                # PDF Processing & Vectorstore Creation
├── requirements.txt         # Project Dependencies
├── .env                     # API Keys (Local Only)
├── .gitignore               # Files to ignore (venv, .env, etc.)
├── data/                    # Folder for Medical PDFs
└── vectorstore/             # Saved FAISS Index
