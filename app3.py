import os
import streamlit as st
import google.generativeai as genai
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Read the Gemini API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini SDK
genai.configure(api_key=GOOGLE_API_KEY)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Meeting GenAI Chatbot", layout="wide")

st.title("📄 Meeting Assistant Chatbot")
st.write("Ask questions based ONLY on the uploaded meeting transcript PDF.")

# -------------------- Sidebar --------------------
st.sidebar.title("📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader(
    "Upload meeting transcript (PDF only)",
    type=["pdf"]
)

# -------------------- Functions --------------------
@st.cache_resource
def process_pdf(file_path):
    """Load and split PDF into chunks"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_documents(documents)


@st.cache_resource
def setup_vectorstore(docs):
    """Create FAISS vector store from documents"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def get_gemini_response(input_text, context):
    """
    Sends the context and question to Gemini for analysis,
    and returns the response text.
    """
    # Initialize the Gemini model
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Build the prompt with context
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
{input_text}

Answer:
"""
    
    # Generate response
    response = model.generate_content(prompt)
    
    return getattr(response, "text", "No text returned from Gemini.")


# -------------------- Main Logic --------------------
if uploaded_file:
    with st.spinner("Processing PDF and creating embeddings..."):
        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # Process PDF
        docs = process_pdf(pdf_path)

        # Create Vector Store
        vectorstore = setup_vectorstore(docs)

        # Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        st.sidebar.success("✅ PDF processed successfully")

    # Chat Input
    input_text = st.text_input("💬 Ask a question about the meeting:", key="input")

    # Action button
    submit = st.button("Analyze")

    # When clicked, analyze
    if submit:
        if not input_text:
            st.warning("⚠️ Please enter a question before submitting.")
            st.stop()
        
        with st.spinner("Thinking..."):
            try:
                # Retrieve relevant documents
                relevant_docs = retriever.invoke(input_text)

                # Build context
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # ⭐ Call Gemini using the function (same style as your screenshot)
                response_text = get_gemini_response(input_text, context)

                # Display result
                st.subheader("Analysis Result")
                st.write(response_text)

            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    st.error("🚫 **Quota Exceeded!** You've hit the free tier limit.")
                    st.info("**Solutions:**\n"
                           "1. Wait a few minutes and try again\n"
                           "2. Generate a new API key at https://aistudio.google.com/app/apikey")
                else:
                    st.error(f"An error occurred: {str(e)}")

else:
    st.info("👈 Please upload a PDF from the sidebar to begin.")