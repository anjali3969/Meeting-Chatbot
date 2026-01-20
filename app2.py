import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Meeting GenAI Chatbot")
st.title("📄 Meeting GenAI Chatbot (T5-small, Local)")
st.write("Upload a meeting transcript PDF and ask questions based ONLY on its content.")
with st.sidebar:
    st.header("📂 Upload Meeting PDF")
    uploaded_file = st.file_uploader("📤 Upload a PDF file", type="pdf")


# -------------------------------------------------
# PDF Processing
# -------------------------------------------------
@st.cache_resource
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

@st.cache_resource
def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(docs, embeddings)

@st.cache_resource
def load_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,

        # 🔥 IMPORTANT PARAMETERS
        do_sample=True,          # enable sampling
        temperature=0.1,         # controls randomness (0.6–0.9 best)
        top_p=0.9,               # nucleus sampling
        repetition_penalty=1.2   # prevents same output
    )

    return HuggingFacePipeline(pipeline=pipe)

# -------------------------------------------------
# Main Logic
# -------------------------------------------------
if uploaded_file:
    with st.spinner("📄 Processing PDF and creating embeddings..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # 🔥 Process immediately after upload
        docs = process_pdf(pdf_path)
        vectorstore = create_vectorstore(docs)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        llm = load_t5_model()

        query = st.text_input("💬 Ask a question about the meeting")
        if query:
            with st.spinner("Thinking..."):
                relevant_docs = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                prompt = f"""
                question: {query}
                context: {context}
                """

                answer = llm.invoke(prompt)
                st.success(answer)
else:
    st.info("👈 Upload a meeting PDF from the sidebar to begin.")

