import os
import tempfile
import streamlit as st
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader  #pdf loader
from langchain_text_splitters import RecursiveCharacterTextSplitter  #text splitter
from langchain_community.vectorstores import FAISS   #similarity search
from langchain_huggingface import HuggingFaceEmbeddings  # generating embeddings
from dotenv import load_dotenv  #load env

#ENV SETUP
load_dotenv(override=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # reading key
genai.configure(api_key=GOOGLE_API_KEY)

#STREAMLIT UI
st.set_page_config(page_title="Meeting GenAI Chatbot", layout="wide")
st.title("📄 Meeting Assistant Chatbot")
st.info("Ask questions based ONLY on the uploaded meeting transcript PDF.")

st.sidebar.title("📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader(
    "Upload meeting transcript (PDF only)",
    type=["pdf"]
)

#SESSION STATE INIT
if "vectorstore" not in st.session_state:  #Stores FAISS vector DB
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:  #Stores retriever object for semantic search.
    st.session_state.retriever = None

if "messages" not in st.session_state:  #Stores full chat history
    st.session_state.messages = []

if "pdf_processed" not in st.session_state:  #embeddings are created only once
    st.session_state.pdf_processed = False

#FUNCTIONS
@st.cache_resource   #faster reloads
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()   #Converts PDF into LangChain Document objects
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

@st.cache_resource
def setup_vectorstore(docs):  #text into numerical vectors
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)  #Stores embeddings inside FAISS for similarity search

def get_gemini_response(question, context):
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 512
        }
    )
    prompt = f"""
You are a meeting assistant.

Use ONLY the information provided in the context below to answer the question.
The answer IS present in the context.

If the exact wording is not available, infer the answer from the relevant sentences.
Do NOT use outside knowledge.
Do NOT mention the context explicitly.

Context:
{context}

Question:
{question}

Answer:
"""
    response = model.generate_content(prompt)
    return getattr(response, "text", "No response from Gemini.")

#PDF PROCESSING
if uploaded_file and not st.session_state.pdf_processed:  #Runs only once after upload.
    with st.spinner("🔄 Processing PDF & creating embeddings..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:  #Creates a temporary PDF file.
            tmp_file.write(uploaded_file.read())  #Writes uploaded PDF to disk
            pdf_path = tmp_file.name

        docs = process_pdf(pdf_path)  
        vectorstore = setup_vectorstore(docs) 

        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) #Retrieves top 3 most relevant chunks per query
        st.session_state.pdf_processed = True

    st.sidebar.success("✅ PDF processed & embeddings ready")

#DISPLAY CHAT HISTORY
for msg in st.session_state.messages:  #Loops through entire conversation.
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#CHAT INPUT (ENABLED ONLY AFTER EMBEDDINGS)
if st.session_state.pdf_processed:
    user_input = st.chat_input("💬 Ask a question about the meeting")
else:
    st.info("👈 Upload a PDF to enable chat")
    user_input = None

#CHAT LOGIC
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})  #Saves user message.

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                relevant_docs = st.session_state.retriever.invoke(user_input)
                context = "\n\n".join(doc.page_content for doc in relevant_docs)  #Combines retrieved chunks
                response = get_gemini_response(user_input, context)

            except Exception as e:
                response = f"⚠️ Error: {e}"

            st.markdown(response)

    # Store assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
