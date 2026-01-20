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
```text
MediBot-AI/
├── medibot.py               # Main Streamlit Application UI
├── ingest.py                # Script for processing PDFs & creating FAISS index
├── chat_loop.py             # CLI based chat interface
├── requirements.txt         # List of Python dependencies
├── .env                     # API Keys (Local use only - Ignored by Git)
├── .gitignore               # Configuration for ignoring unnecessary files
├── data/                    # Directory containing Medical PDF documents
└── vectorstore/             # Saved FAISS vector database files
⚙️ Setup & Installation Guide
1. Clone the Repository
Bash
git clone [https://github.com/JigeishuSrivastava/MediBot-AI.git](https://github.com/JigeishuSrivastava/MediBot-AI.git)
cd MediBot-AI
2. Create and Activate Virtual Environment
Bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
3. Install Required Libraries
Bash
pip install -r requirements.txt
4. Configure Environment Variables
Apne project root mein ek .env file banayein aur ye credentials add karein:

Code snippet
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
5. Initialize Knowledge Base (Vectorization)
Apne PDFs ko data/ folder mein rakhein aur niche di gayi command run karein:

Bash
python ingest.py
6. Start the Assistant
Bash
streamlit run medibot.py
🌐 Deployment on Streamlit Cloud
GitHub par apni repository push karein (Ensure karein .env push na ho).

Streamlit Cloud dashboard par jayein aur repository connect karein.

Advanced Settings mein ja kar Secrets section mein apni API keys TOML format mein dalein:

Ini, TOML
GROQ_API_KEY = "gsk_..."
HF_TOKEN = "hf_..."
Deploy par click karein aur aapka MediBot live ho jayega!

📸 UI Customizations
Glassmorphism: Chat bubbles mein 90% opacity white background ka use kiya gaya hai.

Typography: Behtar readability ke liye pure black text (#000000) ka upyog kiya gaya hai.

Spacing: Boxes ke beech 25px ka professional gap rakha gaya hai.

🤝 Contributing
MediBot ek open-source project hai. Agar aap naye features add karna chahte hain, toh feel free to open a Pull Request!

📜 License
This project is licensed under the MIT License.


---

### Final Implementation:
Ise save karne ke baad terminal mein ye commands chalayein taaki GitHub par update dikhne lage:
1. `git add README.md`
2. `git commit -m "MediBot: Comprehensive README with technical details"`
3. `git push origin main`

**Kya aap chahte hain ki main bot ke liye ek "Disclaimer" section bhi add karun jo bataye ki ye sirf education purpose ke liye hai?**
