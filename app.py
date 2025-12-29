import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langgraph.checkpoint.memory import MemorySaver

# ---------------- CONFIG ----------------
PDF_PATH = r"C:\Users\Karthik\Desktop\clg project\Introduction part.pdf"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 50
MODEL_NAME = "gpt-4o-mini"

# Set your API key
os.environ["OPENAI_API_KEY"] = "API_KEY_HERE"

# ---------------- LOAD PDF ----------------
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

# ---------------- SPLIT ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
docs = splitter.split_documents(pages)

# ---------------- VECTOR STORE ----------------
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------- LLM ----------------
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0
)

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_template(
    """
You are an academic assistant.
Explain the chapter STRICTLY using the provided text.
Do NOT summarize the report structure unless explicitly asked.
Do NOT add external information.

Context:
{context}

Question:
{question}

Answer (detailed, exam-oriented, paragraph form):
"""
)


# ---------------- MEMORY ----------------
memory = MemorySaver()

# ---------------- RAG PIPELINE (LCEL) ----------------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ---------------- ASK QUESTIONS ----------------
questions = [
    "Explain Chapter 1 in detail based on the given text"
]

for q in questions:
    answer = rag_chain.invoke(q)
    print("Q:", q)
    print("A:", answer)
    print("-" * 50)
