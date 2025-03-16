#from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from sentence_transformers import SentenceTransformer
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
#from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Use a markdown file from github page
loader = WebBaseLoader("https://brainlox.com/courses/category/technical")

docs = loader.load()
print(docs[0].page_content[:500])


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

#Create a split of the document using the text splitter
splits = text_splitter.split_documents(docs)
# Extract text from Document objects
splits_text = [doc.page_content for doc in splits]



# Load a pre-trained embedding model
#model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")  # Small & fast model
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Example text to encode
texts = ["Hello, how are you?", "This is an AI model."]

# Generate embeddings
#embeddings = model.encode(splits_text)

# Print embedding shape
#print(embeddings.shape)  # (2, 384)





persist_directory = "docs/chroma/"

# ✅ Ensure you have an embedding model
#embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Convert Documents to Text
#texts = [doc.page_content for doc in splits]  # Extract text from Document objects

# ✅ Store in ChromaDB
vectordb = Chroma.from_texts(
    texts=splits_text,
    embedding=model,
    persist_directory=persist_directory
)

# ✅ Print the number of stored vectors
print("Number of stored embeddings:", vectordb._collection.count())


from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# ✅ Load a local model using Transformers
model_pipeline = pipeline("text-generation", model="facebook/opt-1.3b")

# ✅ Define llm using HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=model_pipeline)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

# Pass question to the qa_chain
question = "What are major topics "
result = qa_chain({"query": question})
result["result"]
print(result["result"])