# ChatBot-with-RAG-and-Chat-History

Chatbot with RAG Pipeline and Conversational history
This is a Streamlit app that allows you to upload PDF files, ask questions based on their content, and receive context-aware answers using a conversational RAG (Retrieval-Augmented Generation) system.

## Features

- Upload and process multiple PDF files.
- Embeds content using HuggingFace model (`all-miniLM-L6-v2`).
- Conversational chatbot using LangChain and GROQ LLM (`Gemma2-9b-It`).
- Maintains per-session chat history.
- Reformulates follow-up questions using chat history.
- Retrieves context-relevant chunks using Chroma vector store.

## Requirements

- Python 3.10+
- GROQ API Key
- HuggingFace API Key

## Installation

```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
