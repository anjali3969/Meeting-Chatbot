import os
import streamlit as st
import google.generativeai as genai
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# ENV SETUP
load_dotenv(override=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

#STREAMLIT UI
st.set_page_config(page_title="Meeting GenAI Chatbot", layout="wide")
st.title("📄 Meeting Assistant Chatbot")
st.write("Ask questions based ONLY on the uploaded meeting transcript PDF.")

st.sidebar.title("📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader(
    "Upload meeting transcript (PDF only)",
    type=["pdf"]
)

#FUNCTIONS
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

@st.cache_resource
def setup_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

def get_gemini_response(question, context):
    model = genai.GenerativeModel("gemini-2.5-flash")
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
    return getattr(response, "text", "No text returned from Gemini.")

#MAIN LOGIC
if uploaded_file:
    with st.spinner("Processing PDF and creating embeddings..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        docs = process_pdf(pdf_path)
        vectorstore = setup_vectorstore(docs)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        st.sidebar.success("✅ PDF processed successfully")

    input_text = st.chat_input("💬 Ask a question about the meeting:", key="input")

    if input_text:
        # Show user message
        with st.chat_message("user", avatar="user"):
            st.markdown(input_text)
    
        # Generate assistant response
        with st.spinner("Thinking..."):
            try:
                relevant_docs = retriever.invoke(input_text)
                context = "\n\n".join(doc.page_content for doc in relevant_docs)
                response_text = get_gemini_response(input_text, context)
    
            except Exception as e:
                response_text = f"⚠️ Error: {e}"
    
        # Show assistant message
        with st.chat_message("assistant", avatar="ai"):
            st.markdown(response_text)
else:
    st.info("👈 Please upload a PDF from the sidebar to begin.")
