# Flask Chatbot with LangChain & Hugging Face

This project is a Flask-based chatbot application that utilizes LangChain for retrieval-augmented generation (RAG) and Hugging Face models for text generation and embedding.

## Features
- Uses **GPT-2** for text generation.
- Embedding model: **sentence-transformers/all-MiniLM-L6-v2**.
- Stores and retrieves embeddings using **ChromaDB**.
- Custom prompt for generating concise answers.
- REST API endpoint (`/ask`) for chatbot interaction.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
2. Install Dependencies
Ensure you have Python installed (recommended: Python 3.8+). Then, install the required libraries:

bash
Copy
Edit
pip install flask langchain langchain_huggingface langchain_chroma transformers sentence-transformers
3. Run the Application
bash
Copy
Edit
python app.py
By default, the app runs on http://0.0.0.0:5000.

API Usage
1. Homepage
GET /
Loads the chatbot interface.
2. Ask a Question
POST /ask
Request Body (JSON):
json
Copy
Edit
{
  "question": "What is artificial intelligence?"
}
Response (JSON):
json
Copy
Edit
{
  "answer": "AI is the simulation of human intelligence in machines..."
}
Customization
Modify the Hugging Face model in:
python
Copy
Edit
model_pipeline = pipeline("text-generation", model="gpt2", max_new_tokens=200)
Change the embedding model in:
python
Copy
Edit
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
Adjust the ChromaDB storage path:
python
Copy
Edit
persist_directory = "docs/chroma/"
License
This project is licensed under the MIT License.

Author
Your Name
GitHub: your-username
Email: your.email@example.com