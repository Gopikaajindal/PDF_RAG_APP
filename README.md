# 📄 AI-Powered Document Question Answering App (PDF, TXT, DOCX)

This is a Gradio-based web app that allows users to upload `.pdf`, `.txt`, or `.docx` files and ask questions about their contents using **Google Gemini** through **LangChain**. The app retrieves the most relevant parts of the document using **FAISS vector search** and provides intelligent answers based on context.

## 🔍 Features

- Upload PDF, TXT, or DOCX files
- Uses SentenceTransformer for chunk embedding
- Stores document chunks in FAISS vector index
- Supports context-aware Q&A using Gemini (via LangChain)
- Simple and interactive UI with Gradio

## 🚀 How to Run

```bash
pip install -r requirements.txt
python app.py
