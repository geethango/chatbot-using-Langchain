from flask import Flask, render_template, request, jsonify
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline
print("hello")
app = Flask(__name__)
print("hello2")
# ✅ Load LLM
print("hello3")
model_pipeline = pipeline("text-generation",model="gpt2", max_new_tokens=200)
print("hello4")
llm = HuggingFacePipeline(pipeline=model_pipeline)
print("hello3")
# ✅ Load Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Load ChromaDB
persist_directory = "docs/chroma/"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
print("Number of stored embeddings:", vectordb._collection.count())

from langchain.prompts import PromptTemplate

# ✅ Correctly define the prompt template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Use the following context to answer the question concisely. 
    If you don't know the answer, just say 'I don't know'. 
    Avoid unnecessary repetition. 

    Context: {context}
    Question: {question}
    """
)

# ✅ Define QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 1}),return_source_documents=False,chain_type_kwargs={
        "prompt":custom_prompt
    }
    )

#print(qa_chain)
@app.route("/")
def home():
    """Render the chatbot UI."""
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    """API to get chatbot responses."""
    data = request.json
    query = data.get("question")
    print(query)
    if not query:
        return jsonify({"error": "No question provided"}), 400

    response = qa_chain.invoke(query)
    print(response)
    response = response[:200] + "..." if len(response) > 200 else response
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
