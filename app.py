from flask import Flask, render_template, request, jsonify
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline

app = Flask(__name__)


model_pipeline = pipeline("text-generation",model="gpt2", max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=model_pipeline)


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


persist_directory = "docs/chroma/"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
print("Number of stored embeddings:", vectordb._collection.count())

from langchain.prompts import PromptTemplate


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


qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 1}),return_source_documents=False,chain_type_kwargs={
        "prompt":custom_prompt
    }
    )


@app.route("/")
def home():
    """Render the chatbot UI."""
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    
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
