import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# .env load karein
load_dotenv()

# Purana Vectorstore load karein
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)

llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))

prompt = ChatPromptTemplate.from_template("""
Use the context to answer:
Context: {context}
Question: {input}
""")

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(db.as_retriever(), combine_docs_chain)

print("\n✅ Bot Ready! Purana vectorstore use ho raha hai.")
while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit"]: break
    response = rag_chain.invoke({"input": user_query})
    print(f"AI: {response['answer']}")