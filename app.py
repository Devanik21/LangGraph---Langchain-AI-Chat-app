import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, JSONLoader,
    UnstructuredMarkdownLoader, UnstructuredPowerPointLoader,
    UnstructuredExcelLoader, UnstructuredHTMLLoader, UnstructuredEPubLoader
)
from langgraph.graph import StateGraph, END

# 🌐 Gemini API Key
api_key = st.secrets["GEMINI_API_KEY"]

# Streamlit page setup
st.set_page_config(page_title="📄 Chat with Your Docs (LangGraph + Gemini)", layout="wide")
st.title("📄 Chat with Your Docs using LangGraph + Gemini")
st.markdown("Upload files (PDF, DOCX, TXT, CSV, JSON, MD, PPTX, XLSX, HTML, EPUB) and ask anything about them!")

# Upload section
uploaded_files = st.file_uploader(
    "Upload your documents", 
    type=["pdf", "txt", "docx", "csv", "json", "md", "pptx", "xlsx", "html", "epub"],
    accept_multiple_files=True
)

# LangGraph state schema
class GraphState(dict):
    pass

# Process files
docs = []
if uploaded_files:
    for file in uploaded_files:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            if suffix == "pdf":
                loader = PyPDFLoader(tmp_path)
            elif suffix == "txt":
                loader = TextLoader(tmp_path)
            elif suffix == "docx":
                loader = Docx2txtLoader(tmp_path)
            elif suffix == "csv":
                loader = CSVLoader(file_path=tmp_path)
            elif suffix == "json":
                loader = JSONLoader(file_path=tmp_path, jq_schema=".", text_content=False)
            elif suffix == "md":
                loader = UnstructuredMarkdownLoader(tmp_path)
            elif suffix == "pptx":
                loader = UnstructuredPowerPointLoader(tmp_path)
            elif suffix == "xlsx":
                loader = UnstructuredExcelLoader(tmp_path)
            elif suffix == "html":
                loader = UnstructuredHTMLLoader(tmp_path)
            elif suffix == "epub":
                loader = UnstructuredEPubLoader(tmp_path)
            else:
                continue

            docs.extend(loader.load())
        except Exception as e:
            st.warning(f"❌ Could not load {file.name}: {str(e)}")

    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # LangGraph node: retrieve documents
    def retrieve_node(state: GraphState) -> GraphState:
        query = state["question"]
        results = retriever.get_relevant_documents(query)
        return GraphState({**state, "docs": results})

    # LangGraph node: generate response
    def generate_node(state: GraphState) -> GraphState:
        context = "\n\n".join([doc.page_content for doc in state["docs"]])
        prompt = f"Context:\n{context}\n\nQuestion: {state['question']}"
        response = llm.invoke(prompt)
        return GraphState({**state, "answer": response.content})

    # LangGraph graph
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.set_finish_point("generate")
    app = workflow.compile()

    # Session memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question about your documents:")
    if st.button("Submit") and query:
        result = app.invoke(GraphState({"question": query}))
        answer = result["answer"]
        st.session_state.chat_history.append((query, answer))

    # Display conversation
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Gemini:** {a}")
else:
    st.info("📂 Upload some documents to get started.")
