# Vector.py
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_storage(pdf_path, model_name):
    model = SentenceTransformer(model_name)
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=40,
        is_separator_regex=False
    )
    splits = text_splitter.split_documents(data)
    embd = SentenceTransformerEmbeddings(model_name=model_name)
    db = FAISS.from_documents(splits, embd)
    db.save_local("faiss_index")

def load_vector_storage(model_name):
    embd = SentenceTransformerEmbeddings(model_name=model_name)
    return FAISS.load_local("faiss_index", embd)
