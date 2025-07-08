import os
from typing import List, Dict, Any
from pypdf import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Define constants
DOCS_DIR = "docs"  # Directory where your legal documents are stored
FAISS_INDEX_PATH = "faiss_index.bin"  # Path to save the FAISS index
METADATA_PATH = "metadata.json"  # Path to save chunk metadata


class DocumentProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the DocumentProcessor with a SentenceTransformer model.
        Args:
            model_name (str): The name of the sentence-transformer model to use for embeddings.
        """
        self.embedding_model = SentenceTransformer(model_name)
        # Initialize text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Size of each text chunk
            chunk_overlap=100,  # Overlap between chunks to maintain context
            length_function=len,  # Function to measure chunk length
            add_start_index=True,  # Add start index to metadata
        )

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts text from a PDF file.
        Args:
            pdf_path (str): The path to the PDF file.
        Returns:
            str: The extracted text.
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""  # Extract text from each page
            return text
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""

    def _extract_text_from_docx(self, docx_path: str) -> str:
        """
        Extracts text from a DOCX file.
        Args:
            docx_path (str): The path to the DOCX file.
        Returns:
            str: The extracted text.
        """
        try:
            doc = Document(docx_path)
            return "\n".join(
                [paragraph.text for paragraph in doc.paragraphs]
            )  # Join paragraphs
        except Exception as e:
            print(f"Error extracting text from DOCX {docx_path}: {e}")
            return ""

    def _load_documents(self, docs_dir: str) -> List[Dict[str, str]]:
        """
        Loads documents from the specified directory and extracts text.
        Args:
            docs_dir (str): The directory containing the documents.
        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing 'text' and 'source' (filename).
        """
        documents = []
        for filename in os.listdir(docs_dir):
            filepath = os.path.join(docs_dir, filename)
            if os.path.isfile(filepath):
                text = ""
                if filename.lower().endswith(".pdf"):
                    text = self._extract_text_from_pdf(filepath)
                elif filename.lower().endswith(".docx"):
                    text = self._extract_text_from_docx(filepath)

                if text:
                    documents.append(
                        {"text": text, "source": filename}
                    )  # Store extracted text and source filename
        return documents

    def _chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Chunks the loaded documents into smaller pieces.
        Args:
            documents (List[Dict[str, str]]): List of documents with 'text' and 'source'.
        Returns:
            List[Dict[str, Any]]: List of chunks with 'text', 'source', and 'chunk_id'.
        """
        chunks = []
        for doc in documents:
            # Split the document text into smaller chunks
            split_texts = self.text_splitter.split_text(doc["text"])
            for i, chunk_text in enumerate(split_texts):
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": doc["source"],
                        "chunk_id": f"{doc['source']}_chunk_{i}",  # Unique ID for each chunk
                    }
                )
        return chunks

    def _create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Creates embeddings for the given text chunks.
        Args:
            chunks (List[Dict[str, Any]]): List of chunks with 'text'.
        Returns:
            np.ndarray: A NumPy array of embeddings.
        """
        texts = [chunk["text"] for chunk in chunks]
        # Encode texts to get their embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return np.array(embeddings).astype("float32")  # Ensure float32 for FAISS

    def create_and_save_vector_store(self):
        """
        Loads, chunks, embeds documents, and saves the FAISS index and metadata.
        """
        print(f"Loading documents from {DOCS_DIR}...")
        documents = self._load_documents(DOCS_DIR)
        if not documents:
            print(
                f"No documents found in {DOCS_DIR}. Please place your PDF/DOCX files there."
            )
            return

        print("Chunking documents...")
        chunks = self._chunk_documents(documents)
        if not chunks:
            print("No chunks generated from documents.")
            return

        print("Creating embeddings for chunks...")
        embeddings = self._create_embeddings(chunks)

        # Get the dimension of the embeddings
        dimension = embeddings.shape[1]
        # Create a FAISS index (IndexFlatL2 for L2 distance)
        index = faiss.IndexFlatL2(dimension)
        # Add the embeddings to the index
        index.add(embeddings)

        print(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
        faiss.write_index(index, FAISS_INDEX_PATH)

        # Save chunk metadata (text, source, chunk_id)
        metadata = [
            {"text": c["text"], "source": c["source"], "chunk_id": c["chunk_id"]}
            for c in chunks
        ]
        print(f"Saving metadata to {METADATA_PATH}...")
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        print("Vector store created and saved successfully.")

    def load_vector_store(self):
        """
        Loads the FAISS index and metadata from disk.
        Returns:
            tuple: A tuple containing the loaded FAISS index and metadata.
        """
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
            print(
                "FAISS index or metadata not found. Please run document processing first."
            )
            return None, None

        print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
        index = faiss.read_index(FAISS_INDEX_PATH)

        print(f"Loading metadata from {METADATA_PATH}...")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return index, metadata


if __name__ == "__main__":
    # This block runs when document_processor.py is executed directly
    # It will process your documents and create the FAISS index
    processor = DocumentProcessor()
    processor.create_and_save_vector_store()
