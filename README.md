# Maintenance Manual RAG Assistant

A small web app that lets you **ask questions about PDF maintenance manuals** using a Retrieval-Augmented Generation (RAG) pipeline.

You upload a PDF manual, the app builds a vector index over its text, and then an OpenAI model (`gpt-4.1-mini`) answers your questions using the most relevant sections of the document.

> **Note:** At the moment this project supports PDFs that contain **selectable text** (i.e. you can highlight text in a PDF viewer). Support for scanned/image-only PDFs via OCR is planned but not implemented yet.

---

## Features

- 📝 Upload a PDF maintenance / service manual
- 🔍 Automatic text extraction and chunking
- 📐 Embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`)
- 📚 Vector search using FAISS to find relevant chunks
- 🤖 Answer generation using OpenAI’s `gpt-4.1-mini`
- 🖥 Simple Streamlit web UI:
  - “Index manual” button to build the vector index
  - Text box to ask natural-language questions
  - Optional view of the retrieved chunks used to answer

---

## How it works (high level)

1. **Upload** a PDF through the Streamlit UI.
2. The app **extracts text** from the PDF (using `pypdf`).
3. The text is **split into overlapping chunks**.
4. Each chunk is **embedded** into a vector using a SentenceTransformer model.
5. The vectors are stored in a **FAISS index**.
6. When you ask a question:
   - The question is embedded.
   - The index returns the **top-k most similar chunks**.
   - Those chunks + your question are passed to an OpenAI chat model (`gpt-4.1-mini`).
   - The model returns a **step-by-step answer** grounded in the manual.

---

## Prerequisites

To run this project locally, you’ll need:

- **Python 3.10+**
- **pip** (Python package manager)
- An **OpenAI API key** with an active billing/quota (ChatGPT Plus ≠ API credits)
- (Optional but recommended) **virtual environment** support (`venv`)

The instructions below are written for Windows/PowerShell, but are easy to adapt for macOS/Linux.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

### 2. Create and activate a virtual environment (instructions for windows)
pyhton -m venv .venv
.venv\Scripts\Activate.ps1

### For use in Linux use the following instead
python -m venv .venv
source .venv/bin/activate

### 3. Install the required dependencies
pip install -r requirements.txt

### 4. Set up your OpenAI key
### This will require you to create a file name ".env" in the project in the same folder as "app.py"
### NOTE: this step does require that you have billing/quotos enabled through OpenAI for API usauge. 
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

### 5. Run the app
### With the virtual environment active and ".env" set up run:
streamlight run app.py

### Limitations and future planned work
### 1. This iteration only supports PDFs with a text layer (selectable text), does not work with PDFs that are images and do not have selectable text
### 2. Large PDFs may take some time to index in the applet
### 3. The FAISS index is currently in-memory only, it does not write to permenant storage with this iteration, so the index will have to be rebuilt each time the app is closed
### Planned changes in future iteration
### 1. Implement OCR to allow for the app to handle scanned or image only PDFs
### 2. Save and Load the FAISS indices and chunks to prevent having to reindex PDFs on every app load
### 3. Add a token usage meter to allow users to monitor API token usage to help prevent possible overruns 

