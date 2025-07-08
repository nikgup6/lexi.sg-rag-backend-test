import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import json
import os  # Import the os module
from typing import List, Dict, Any

# Define constants for file paths
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.json"


class RAGPipeline:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "llama2",
    ):
        """
        Initializes the RAGPipeline with an embedding model and LLM model.
        Args:
            embedding_model_name (str): Name of the sentence-transformer model for embeddings.
            llm_model_name (str): Name of the Ollama model to use for generation (e.g., 'llama2').
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm_model_name = llm_model_name
        self.faiss_index = None
        self.metadata = None
        # Get Ollama host from environment variable, default to localhost
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._load_vector_store()  # Load the pre-built vector store on initialization

    def _load_vector_store(self):
        """
        Loads the FAISS index and metadata from disk.
        """
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
            print(
                "Error: FAISS index or metadata not found. Please run document_processor.py first."
            )
            return

        try:
            self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print("FAISS index and metadata loaded successfully.")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.faiss_index = None
            self.metadata = None

    def retrieve_documents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant document chunks from the FAISS index.
        Args:
            query (str): The natural language query.
            k (int): The number of top relevant chunks to retrieve.
        Returns:
            List[Dict[str, Any]]: A list of retrieved chunks, each with 'text', 'source', and 'chunk_id'.
        """
        if self.faiss_index is None or self.metadata is None:
            print("Vector store not loaded. Cannot retrieve documents.")
            return []

        # Encode the query into an embedding
        query_embedding = self.embedding_model.encode([query]).astype("float32")

        # Perform a similarity search in the FAISS index
        # D: distances, I: indices of the nearest neighbors
        distances, indices = self.faiss_index.search(query_embedding, k)

        retrieved_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Ensure the index is valid
                chunk_data = self.metadata[idx]
                retrieved_chunks.append(
                    {
                        "text": chunk_data["text"],
                        "source": chunk_data["source"],
                        "chunk_id": chunk_data["chunk_id"],
                    }
                )
        return retrieved_chunks

    def generate_answer(
        self, query: str, context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates an answer using the LLM based on the query and retrieved context.
        Args:
            query (str): The original natural language query.
            context_chunks (List[Dict[str, Any]]): List of retrieved document chunks.
        Returns:
            Dict[str, Any]: A dictionary containing the generated 'answer' and 'citations'.
        """
        if not context_chunks:
            return {
                "answer": "I could not find relevant information in the documents to answer your query.",
                "citations": [],
            }

        # Construct the context string from retrieved chunks
        context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])

        # Create the prompt for the LLM
        prompt = f"""
        You are a helpful AI assistant specialized in legal information.
        Based on the following context, answer the user's query concisely and accurately.
        If the answer cannot be found in the context, state that you don't have enough information.

        Context:
        {context_text}

        User Query: {query}

        Answer:
        """

        try:
            # Call the Ollama LLM API, using the configured host
            response = ollama.chat(
                model=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                host=self.ollama_host,  # Use the configured host
            )
            generated_answer = response["message"]["content"].strip()

            # Prepare citations
            citations = []
            for chunk in context_chunks:
                citations.append({"text": chunk["text"], "source": chunk["source"]})

            return {"answer": generated_answer, "citations": citations}
        except ollama.ResponseError as e:
            print(f"Error communicating with Ollama: {e}")
            return {
                "answer": f"An error occurred while generating the answer (Ollama connection issue: {e}). Please ensure Ollama server is running and the '{self.llm_model_name}' model is available at {self.ollama_host}.",
                "citations": [],
            }
        except Exception as e:
            print(f"An unexpected error occurred during answer generation: {e}")
            return {
                "answer": f"An unexpected error occurred while generating the answer: {e}",
                "citations": [],
            }

    def query_rag_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Runs the full RAG pipeline: retrieve and then generate.
        Args:
            query (str): The natural language query.
        Returns:
            Dict[str, Any]: The generated answer and citations.
        """
        print(f"Processing query: '{query}'")
        retrieved_chunks = self.retrieve_documents(query)
        print(f"Retrieved {len(retrieved_chunks)} chunks.")
        response = self.generate_answer(query, retrieved_chunks)
        return response


# Example usage (for testing purposes, not part of the FastAPI app directly)
if __name__ == "__main__":
    # Ensure you have run document_processor.py first to create the index and metadata
    # And ensure Ollama server is running and 'llama2' model is pulled:
    # 1. Download Ollama: https://ollama.com/download
    # 2. Run in terminal: `ollama run llama2` (this downloads the model)
    # 3. Keep the Ollama server running in the background.

    rag_pipeline = RAGPipeline()

    if rag_pipeline.faiss_index and rag_pipeline.metadata:
        sample_query = "Is an insurance company liable to pay compensation if a transport vehicle involved in an accident was being used without a valid permit?"
        result = rag_pipeline.query_rag_pipeline(sample_query)
        print("\n--- RAG Result ---")
        print(f"Answer: {result['answer']}")
        print("Citations:")
        for citation in result["citations"]:
            print(
                f"  - Source: {citation['source']}, Text: '{citation['text'][:100]}...'"
            )
    else:
        print("RAG pipeline not ready. Please check previous error messages.")
