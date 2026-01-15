"""
Docstring for src.rag_manual
Core RAG engine for maintenance manuals.

This module defines the ManualRAG class, which is responsible for:
-Extracting text from a PDF
-Splitting that text into overlapping chunks
-Encoding chunks as vectors using a sentence-transformer embedding model
-Building a FAISS index over those vectors 
-Retrieving the most relevant chunks for a given user query
"""
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


class ManualRAG:
    """
    ManualRAG encapsulates the Retrieval part of a RAG system

    It DOES NOT call the LLM directly. Instead, it:
    -Reads a PDF
    -Prepares an embedding index from the text
    -Retrieves top-k chunks for a query

    The LLM call happens in src/llm_answer.py, which takes these chunks
    and generates a natural language answer. 
    """
    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding model and empty index."""
        #Load the embedding model once. This can be relatively expensive,
        #which is why we create the ManualRAG instance once and cache it in app. 
        self.embed_model = SentenceTransformer(embed_model_name)
        #Will hold the FAISS index once built.
        self.index: faiss.Index | None = None
        #Will store the list of text chunks corresponding to the index vectors.
        self.chunks: list[str] | None = None

    def extract_text_from_pdf(self, file_obj_or_path) -> str:
        """
        Extract text from a PDF.
        Accepts either a filesystem path or a file-like object (e.g. Streamlit UploadedFile).
        """
        #Handle both "path-like" and "file-like" inputs.
        if isinstance(file_obj_or_path, (str, Path)):
            reader = PdfReader(str(file_obj_or_path))
        else:
            reader = PdfReader(file_obj_or_path)

        pages_text: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        #Join all pages with blank lines between them. 
        return "\n\n".join(pages_text)

    @staticmethod
    def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
        """
        Split the full manual text into overlapping charcater based chunks.

        This is a simple first pass, better chunking methods could be used,
        such as by paragraphs, headings, etc. if required/to acheive better results

        :param text: the raw text of the manual
        :param max_chars: maximum number of characters per chunk.
        :param overlap: Number of characters to overlap between consecutive chunks,
                        to avoid cutting important sentences in half.
        :return: List of chunk strings.                 
        """
        chunks: list[str] = []
        start = 0
        n = len(text)
        #Slide a window of max_chars through the text, with overlap
        while start < n:
            end = start + max_chars
            chunk = text[start:end]
            chunks.append(chunk)
            #Move the window forward, but step back by 'overlap' so chunks
            # share some context boundry
            start = end - overlap

        return chunks

    def build_index_from_pdf(self, file_obj_or_path) -> None:
        """
        Build the FAISS index from a PDF.

        Steps:
        1. Extract text from a PDF
        2. Chunk the text
        3. Compute embeddings for each chunk
        4. Build a FAISS index over those embeddings

        :param file_obj_or_path: PDF path or file-like object
        :raises ValueError: if no text can be extracted
        """

        #Step 1. extract raw text from the PDF
        text = self.extract_text_from_pdf(file_obj_or_path)

        # Throws ValueError if PDF is not pure text
        #Will create another version that incorporates OCR to resolve this issue
        if not text or not text.strip():
            raise ValueError("No extractable text found in the PDF (it may be scanned as images only).")

        #Step 2. Chunk the text
        chunks = self.chunk_text(text)
        if not chunks:
            # Fallback in case chunk_text returns an empty list for some reason
            chunks = [text]
        #Save chunks so they can be returned during retrieval
        self.chunks = chunks

        #Step 3. Compute the embeddings for each chunk using the sentence-transformer model
        embeddings = self.embed_model.encode(chunks, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")

        # If we got a single 1-D vector (dim,), make it (1, dim)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def retrieve_chunks(self, query: str, k: int = 5) -> list[str]:
        """
        Retrieve the top-K most relevant text chunks for a given query.

        :param query: User's natural-language question
        :param k: Number of chunks to retrieve
        :return: List of chunk strings, ordered by similarity
        "raises RuntimeError: if build_index_from_pdf hasnt been called yet.
        """

        #Make sure something exist to be indexed
        if self.index is None or self.chunks is None:
            raise RuntimeError("Index not built. Call build_index_from_pdf first.")
        #Embed the query into the same vector as the chunks.
        q_emb = self.embed_model.encode([query]).astype("float32")
        #Search the FAISS index for the k nearest chunk vectors.
        distances, indices = self.index.search(q_emb, k)
        #indices is a 2D array, guard against any out of range indices
        return [self.chunks[i] for i in indices[0] if 0 <= i < len(self.chunks)]