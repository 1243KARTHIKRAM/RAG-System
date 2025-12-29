# RAG NLP – PDF Question Answering System

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain**, **FAISS**, and **OpenAI LLMs** to answer questions strictly based on the content of a provided PDF document.

The system is designed mainly for **academic and exam-oriented use**, ensuring that answers are generated **only from the given PDF text**, without adding external or hallucinated information.

---

## 🚀 Features

- Load and process PDF documents
- Split text into overlapping chunks for better contextual understanding
- Generate vector embeddings using OpenAI Embeddings
- Store and retrieve document chunks using FAISS vector database
- Answer questions using an LLM with retrieved context
- Strictly follows the source document (no external knowledge)
- Suitable for college projects and viva demonstrations

---

## 🛠️ Tech Stack

- Python
- LangChain
- LangGraph
- FAISS
- OpenAI API
- PyPDFLoader

---


---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/rag-nlp.git
cd rag-nlp

langchain
langchain-community
langchain-openai
langchain-text-splitters
langgraph
faiss-cpu
pypdf
openai


