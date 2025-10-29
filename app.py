import os
from dotenv import load_dotenv

# --- 1. Load Environment Variables ---
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY not found in .env file")
    exit()
print("API Key Loaded.")

# --- 2. Corrected LangChain Imports ---
# These imports are now correct.

# Updated imports for LangChain v1.x RAG pattern
try:
    from langchain_community.llms import HuggingFaceHub
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    print(f"ImportError: {e}")
    print("One or more LangChain packages are missing or outdated.")
    print("Please run this command in your terminal to fix it:")
    print("pip install -r requirements.txt")
    exit()

print("All libraries loaded successfully.")

# --- 3. Initialize LLM and Embeddings Model ---
# Use HuggingFaceHub if token present; otherwise use local transformers pipeline as a fallback
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if HUGGINGFACE_TOKEN:
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 256},
        huggingfacehub_api_token=HUGGINGFACE_TOKEN,
    )
else:
    # Local fallback using transformers pipeline (no token required)
    from transformers import pipeline

    class LocalLLM:
        def __init__(self, model="google/flan-t5-base"):
            # text2text-generation works well for instruction-style models
            self.pipe = pipeline("text2text-generation", model=model)

        def invoke(self, prompt: str):
            out = self.pipe(prompt, max_length=256, do_sample=False)
            # pipeline returns a list of dicts with 'generated_text'
            return out[0]["generated_text"]

    llm = LocalLLM()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("LLM and Embeddings Loaded (HuggingFaceHub or local fallback).")

# --- 4. Load Your Data ---
try:
    loader = TextLoader("my_data.txt")
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s).")
except FileNotFoundError:
    print("Error: 'my_data.txt' not found.")
    print("Please create this file in the same directory and add some text to it.")
    exit()

# --- 5. Split the Data into Chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)
print(f"Split into {len(split_docs)} chunks.")

# --- 6. Create Embeddings and Store in Vector Database (FAISS) ---
print("Creating vector store...")
db = FAISS.from_documents(split_docs, embeddings)
print("Vector Store Created.")

# --- 7. Create the Retriever ---
retriever = db.as_retriever()

# --- 8. Create the Prompt Template ---
prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}
""")



# --- 9. Manual RAG Chain ---
def ask_question(question):
    # Retrieve relevant documents
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = prompt_template.format(context=context, input=question)
    response = llm.invoke(prompt)
    return response

print("Manual RAG Chain Ready.")

# --- 10. Ask a Question! ---

print("---")
question = "What is a Flibbertigibbet and where does it live?"
print(f"Question: {question}")
response = ask_question(question)
print("\nAnswer:")
print(response)