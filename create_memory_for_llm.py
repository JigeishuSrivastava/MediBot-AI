import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# .env load karein taaki LangSmith tracking enable ho sake
load_dotenv()

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def ingest_data():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} folder nahi mila.")
        return
    
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    db = FAISS.from_documents(text_chunks, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ Naya Vector DB save ho gaya hai: {DB_FAISS_PATH}")

if __name__ == "__main__":
   
    pass