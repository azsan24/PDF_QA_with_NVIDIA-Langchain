import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
api_key = os.getenv("nvidia_api_key")

# Initialize model
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", api_key=api_key)

# UI Title
st.title("📄 Ask Questions About PDF Documents")

# Prepare prompt template
prompt_template = ChatPromptTemplate.from_template("""
Answer the question based only on the provided context.
<context>
{context}
</context>
Question: {input}
""")

# Load and embed documents
def embed_documents():
    if "vectors" not in st.session_state:
        loader = PyPDFDirectoryLoader("./us_census")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        embeddings = NVIDIAEmbeddings()
        st.session_state.vectors = FAISS.from_documents(chunks, embeddings)
        st.write("✅ Document embeddings created and stored in FAISS.")

# Button to trigger embedding
if st.button("📚 Load & Embed Documents"):
    embed_documents()

# Input field for question
question = st.text_input("❓ Enter your question:")

# Answer question
if question:
    if "vectors" not in st.session_state:
        st.warning("Please load the documents first.")
    else:
        retriever = st.session_state.vectors.as_retriever()
        doc_chain = create_stuff_documents_chain(llm, prompt_template)
        qa_chain = create_retrieval_chain(retriever, doc_chain)

        start = time.process_time()
        response = qa_chain.invoke({'input': question})
        elapsed = time.process_time() - start

        st.success(f"Answer: {response['answer']} (in {elapsed:.2f} seconds)")

        with st.expander("📂 Relevant Document Snippets"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.markdown("---")
