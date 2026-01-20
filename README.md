# 🏥 MediBot: Advanced AI-Powered Medical Assistant (RAG Pipeline)

MediBot ek professional **Generative AI** application hai jo medical knowledge base (PDF documents) aur Large Language Models (LLMs) ka upyog karke user ke sawalon ka satik aur context-aware jawab deta hai. Yeh project **Retrieval-Augmented Generation (RAG)** architecture par adharit hai.

---

## 🌟 Key Highlights
* **Intelligent Document Retrieval**: FAISS (Facebook AI Similarity Search) ka upyog karke hazaron medical pages mein se seconds mein relevant context dhoondta hai.
* **Ultra-Fast Inference**: Groq Cloud inference engine ke saath **Llama-3.1-8b** model ka use karke super-fast responses generate karta hai.
* **Professional Medical UI**: Streamlit par banaya gaya ek custom interface jisme glass-morphism design aur black text readability ka dhyan rakha gaya hai.
* **Contextual Accuracy**: Bot sirf provide kiye gaye medical context se hi jawab deta hai, jisse AI hallucinations ka khatra kam ho jata hai.

---

## 🏗️ Technical Architecture & Workflow

MediBot ka internal workflow niche diye gaye steps par kaam karta hai:

1.  **Data Extraction**: `PyPDFLoader` ka upyog karke `data/` folder se medical PDFs ko read kiya jata hai.
2.  **Smart Chunking**: Bade documents ko `RecursiveCharacterTextSplitter` ke zariye 500 characters ke chhote chunks mein toda jata hai taaki LLM context window mein fit ho sake.
3.  **Vector Embeddings**: Hugging Face ka `all-MiniLM-L6-v2` model har chunk ko high-dimensional vector embeddings mein convert karta hai.
4.  **Vector Storage**: In embeddings ko local FAISS database mein store kiya jata hai fast similarity search ke liye.
5.  **RAG Execution**: User ki query aane par, retriever top-3 relevant chunks nikaalta hai aur use Prompt Template ke saath Groq LLM ko bhejta hai.



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

Is project ka directory structure niche diya gaya hai taaki aap component flow ko samajh sakein:

```text
MediBot-AI/
├── data/                        # Input directory (Yahan apni Medical PDFs rakhein)
│   └── medical_manual.pdf       # Sample medical document
├── vectorstore/                 # Vector Database storage
│   └── db_faiss/                # FAISS index files (index.faiss, index.pkl)
├── venv/                        # Python Virtual Environment (Local only)
├── .env                         # Environment variables (API Keys - Hidden)
├── .gitignore                   # Git ko batane ke liye ki kaunsi files push nahi karni
├── requirements.txt             # Python libraries ki list (Deployment ke liye zaroori)
├── README.md                    # Project documentation
├── ingest.py                    # Data processing script (PDF to Vectorstore)
├── medibot.py                   # Main Application file (Streamlit UI)
└── chat_loop.py                 # Backend testing script (Terminal-based chat)
