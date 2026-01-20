# 🏥 MediBot: Advanced AI-Powered Medical Assistant (RAG Pipeline)

MediBot is a professional **Generative AI** application that leverages a custom medical knowledge base (PDF documents) and Large Language Models (LLMs) to provide accurate, context-aware answers to user queries. The project is built on the **Retrieval-Augmented Generation (RAG)** architecture.

---

## 🌟 Key Highlights
* **Intelligent Document Retrieval**: Uses FAISS (Facebook AI Similarity Search) to find relevant context from thousands of medical pages within seconds.
* **Ultra-Fast Inference**: Utilizes the **Llama-3.1-8b** model via the Groq Cloud inference engine for lightning-fast response generation.
* **Professional Medical UI**: Developed with Streamlit, featuring a custom glass-morphism design and high-readability typography.
* **Contextual Accuracy**: The bot strictly answers based on the provided medical context, significantly reducing the risk of AI hallucinations.

---

## 🏗️ Technical Architecture & Workflow

MediBot's internal workflow follows a structured RAG pipeline:



1.  **Data Extraction**: Uses `PyPDFLoader` to read medical PDFs from the `data/` directory.
2.  **Smart Chunking**: Large documents are split into smaller 500-character segments using `RecursiveCharacterTextSplitter` to fit within the LLM's context window.
3.  **Vector Embeddings**: The Hugging Face `all-MiniLM-L6-v2` model converts each chunk into high-dimensional vector embeddings.
4.  **Vector Storage**: Embeddings are stored in a local FAISS database for high-speed similarity searches.
5.  **RAG Execution**: When a user submits a query, the retriever fetches the top 3 relevant chunks and sends them along with a custom Prompt Template to the Groq LLM.

---

## 🛠️ Tech Stack & Dependencies

| Component | Technology Used |
| :--- | :--- |
| **Programming Language** | Python 3.10+ |
| **Frontend Framework** | Streamlit |
| **Orchestration Layer** | LangChain |
| **LLM Model** | Groq (Llama-3.1-8b-instant) |
| **Embedding Model** | Hugging Face (sentence-transformers) |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |

---

## 📂 Project Structure

The directory structure is organized to ensure a clean separation of concerns and data flow:

```text
MediBot-AI/
├── data/                        # Input directory (Place your Medical PDFs here)
│   └── medical_manual.pdf       # Sample medical document
├── vectorstore/                 # Vector Database storage
│   └── db_faiss/                # FAISS index files (index.faiss, index.pkl)
├── venv/                        # Python Virtual Environment
├── .env                         # Environment variables (API Keys - Git ignored)
├── .gitignore                   # Files to be excluded from version control
├── requirements.txt             # List of Python libraries for deployment
├── README.md                    # Project documentation
├── ingest.py                    # Data processing script (PDF to Vectorstore)
├── medibot.py                   # Main Application file (Streamlit UI)
└── chat_loop.py                 # Backend testing script (Terminal-based chat)
