lexi.sg-rag-backend-test
This repository contains a Retrieval-Augmented Generation (RAG) backend service designed to answer natural language legal queries by retrieving information from a corpus of legal documents and generating concise, cited answers.

Objective
The service accepts a natural language legal query and returns:

A generated answer based on document retrieval.

A list of citations from the original documents (text + source).

Stack
Backend Framework: FastAPI (Python)

Document Processing: pypdf, python-docx, langchain (for text splitting)

Embeddings: sentence-transformers (all-MiniLM-L6-v2)

Vector Database: FAISS

Local LLM: Ollama (with llama2 model)

Requirements
Python 3.8+

Ollama (for running the local LLM)

Setup Instructions
Follow these steps to set up and run the RAG backend service:

1. Clone the Repository
   git clone https://github.com/your-username/lexi.sg-rag-backend-test.git
   cd lexi.sg-rag-backend-test

2. Create a Virtual Environment and Install Dependencies
   It's highly recommended to use a virtual environment.

python -m venv venv

# On Windows

.\venv\Scripts\activate

# On macOS/Linux

source venv/bin/activate

pip install -r requirements.txt

3. Download Sample Legal Documents
   Create a docs directory inside the lexi.sg-rag-backend-test folder and place your sample PDF and DOCX legal documents there.

lexi.sg-rag-backend-test/
├── app/
├── docs/
│ ├── sample_legal_doc_1.pdf
│ ├── sample_legal_doc_2.docx
│ └── ...
├── requirements.txt
├── README.md

You can download sample PDFs and DOCX files from the Lexi Legal Docs Repository.

4. Set up Ollama
   The project uses Ollama to run a local LLM (e.g., llama2).

Download and Install Ollama:
Follow the instructions on the official Ollama website: https://ollama.com/download

Pull the llama2 Model:
Once Ollama is installed, open your terminal and run:

ollama pull llama2

This will download the llama2 model. Ensure the Ollama server is running in the background before proceeding.

5. Process Documents and Build the Vector Store
   Navigate to the app directory and run the document_processor.py script. This will extract text, chunk documents, create embeddings, and save the FAISS index and metadata.

cd app
python document_processor.py

You should see output indicating document loading, chunking, embedding creation, and saving of faiss_index.bin and metadata.json files in the root directory (lexi.sg-rag-backend-test/).

6. Run the FastAPI Backend
   After the vector store is successfully created, you can start the FastAPI application.
   Make sure you are in the root directory of your project (lexi.sg-rag-backend-test/).

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

The --reload flag is useful for development as it automatically restarts the server on code changes. The --host 0.0.0.0 makes the server accessible from other devices on your network (if needed), and --port 8000 sets the port.

You should see output similar to:

INFO: Will watch for changes in these directories: ['/path/to/lexi.sg-rag-backend-test']
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO: Started reloader process [PID]
INFO: Started server process [PID]
INFO: Waiting for application startup.
INFO: Application startup complete.

How to Test the API
Once the FastAPI server is running, you can test the /query endpoint using curl, Postman, Insomnia, or by accessing the interactive API documentation.

Using FastAPI's Interactive Docs (Recommended)
Open your web browser and go to:
http://127.0.0.1:8000/docs

You will see the Swagger UI, where you can interact with the /query endpoint.

Click on /query -> POST.

Click "Try it out".

Modify the request body with your query.

Example Input/Output
Request Body (JSON):

{
"query": "Is an insurance company liable to pay compensation if a transport vehicle involved in an accident was being used without a valid permit?"
}

Example Response (JSON):

{
"answer": "No, an insurance company is not liable to pay compensation if a transport vehicle is used without a valid permit at the time of the accident. The Supreme Court held that use of a vehicle in a public place without a permit is a fundamental statutory infraction, and such a situation is not equivalent to cases involving absence of licence, fake licence, or breach of conditions such as overloading. Therefore, the insurer is entitled to recover the compensation amount from the owner and driver after paying the claim.",
"citations": [
{
"text": "Use of a vehicle in a public place without a permit is a fundamental statutory infraction. The said situations cannot be equated with absence of licence or a fake licence or a licence for different kind of vehicle, or, for that matter, violation of a condition of carrying more number of passengers.",
"source": "Doc_Name.docx"
},
{
"text": "Therefore, the tribunal as well as the High Court had directed that the insurer shall be entitled to recover the same from the owner and the driver.",
"source": "Doc_Name.docx"
}
]
}

(Note: Doc_Name.docx will be replaced by the actual filename of your document that contains the relevant text.)

Bonus Points: Hosting the Solution
For hosting, you could consider platforms like:

Render: Offers free tiers for web services.

Vercel/Netlify: Excellent for frontend, but can host serverless functions for backend.

Google Cloud Run / AWS Fargate: Serverless container platforms.

Hugging Face Spaces: Can host Gradio/Streamlit apps, and also custom Docker images.

For Ollama, you would typically need a virtual machine or a dedicated server where Ollama can run. Some cloud providers offer GPU instances which would be beneficial for LLMs. Alternatively, for a purely serverless setup, you might explore cloud-hosted LLM APIs (though the prompt specified free/open-source local LLMs).
