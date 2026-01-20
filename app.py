import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="Meeting GenAI Chatbot")
st.title("📄 Meeting Assistant Chatbot")

# Load PDF
@st.cache_resource
def load_pdf():
    loader = PyPDFLoader("meeting.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

docs = load_pdf()

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Vector Store
vectorstore = FAISS.from_documents(docs, embeddings)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# RAG Chain
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)

# Chat UI
query = st.text_input("Ask a question about the meeting:")

if query:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke({"query": query})
        st.success(response)
