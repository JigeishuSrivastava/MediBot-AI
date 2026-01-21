import os
import streamlit as st
from dotenv import load_dotenv

# Local environment variables load karne ke liye
load_dotenv()

# --- LangSmith Config (Local aur Live dono ke liye) ---
# Hum .env file se keys uthayenge
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "MediBot-Project")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Vector Database Path
DB_FAISS_PATH = "vectorstore/db_faiss"

# --- Page Configuration ---
st.set_page_config(page_title="MediBot - AI Medical Assistant", page_icon="🏥", layout="centered")

@st.cache_resource
def get_vectorstore():
    """FAISS Vectorstore load karne ke liye"""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    if not os.path.exists(DB_FAISS_PATH):
        st.error(f"Error: Vectorstore path '{DB_FAISS_PATH}' nahi mila!")
        return None
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt():
    """Medical Instructions ke liye custom prompt template"""
    prompt_template = """
    Use the pieces of information provided in the context to answer the user's medical question.
    If you don't know the answer, just say that you don't know. 
    Don't provide anything outside of the given context.
    
    Context: {context}
    Question: {question}
    
    Start the answer directly. Keep it professional and concise.
    """
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def main():
    # --- ORIGINAL BACKGROUND & CSS (AS PER YOUR REQUEST) ---
    medical_style = """
    <style>
    /* Background setup with your original URL */
    .stApp {
        background-image: url("https://wallpaperaccess.com/full/624111.jpg");
        background-attachment: fixed;
        background-size: cover;
    }
    
    /* Main Chat Container spacing */
    .stChatMessageContainer {
        gap: 25px !important; 
        padding-top: 20px !important;
    }

    /* Chat messages box style (Original White Boxes) */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #000000 !important;
        border-radius: 15px !important;
        border: 1px solid rgba(0, 80, 150, 0.2) !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2) !important;
        margin-bottom: 25px !important;
    }

    /* Message text black and clear */
    .stChatMessage p, .stChatMessage li {
        color: #000000 !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
    
    /* Title box styling */
    h1 {
        color: #000000 !important;
        background: rgba(255, 255, 255, 0.85);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 40px !important;
    }

    /* Input Box text color fix */
    .stChatInput textarea {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    .stChatInput textarea::placeholder {
        color: rgba(255, 255, 255, 0.7) !important;
    }

    .stChatInputContainer {
        padding-bottom: 30px !important;
        background-color: transparent !important;
    }
    </style>
    """
    st.markdown(medical_style, unsafe_allow_html=True)
    
    st.title("🛡️ MediBot: AI Medical Assistant")

    # API Key check from .env
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY missing! .env check karein.")
        return

    # Chat Session State
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # User Input
    user_input = st.chat_input("Write your Medical Questions Here")

    if user_input:
        st.chat_message('user').markdown(user_input)
        st.session_state.messages.append({'role': 'user', 'content': user_input})

        try:
            vectorstore = get_vectorstore()
            if vectorstore:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key),
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3})
                )

                with st.spinner("Searching medical database..."):
                    response = qa_chain.invoke({'query': user_input})
                    answer = response["result"]

                with st.chat_message('assistant'):
                    st.markdown(answer)
                
                st.session_state.messages.append({'role': 'assistant', 'content': answer})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()