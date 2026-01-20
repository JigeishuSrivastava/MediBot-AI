import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# ---------------------------------------------------
# Load environment variables
# ---------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")

# ---------------------------------------------------
# Setup LLM (Groq)
# ---------------------------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=512,
    groq_api_key=GROQ_API_KEY
)

# ---------------------------------------------------
# Prompt Template
# ---------------------------------------------------
system_prompt = (
    "You are a helpful AI assistant. "
    "Use the following retrieved context to answer the question. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# ---------------------------------------------------
# Load FAISS Vector DB
# ---------------------------------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# ---------------------------------------------------
# Build RAG Chain
# ---------------------------------------------------
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(
    retriever,
    combine_docs_chain
)

# ---------------------------------------------------
# Chat loop
# ---------------------------------------------------
print("\n✅ GenAI RAG Bot Started (type 'exit' to quit)\n")

while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit"]:
        break

    response = rag_chain.invoke({"input": user_query})

    print("\nAI:", response["answer"])
    print("\n--- Sources ---")
    for i, doc in enumerate(response["context"], 1):
        print(f"{i}. {doc.metadata}")
