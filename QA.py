# QA.py
import os
import google.generativeai as palm
from Vector import load_vector_storage, create_vector_storage


# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


def answer_question(pdf_path, question):
    # Load or create vector storage
    model_name = "all-MiniLM-L6-v2"
    if not os.path.exists("faiss_index"):
        create_vector_storage(pdf_path, model_name)
    db = load_vector_storage(model_name)
    

    # Configure Google API
    google_api_key = os.getenv('GOOGLE_API_KEY')
    palm.configure(api_key=google_api_key)


    # Define the query
    query = question
   
    # Integrate retriever
    retriever = db.as_retriever()
    try:
        docs = retriever.invoke(query)
        if docs:
            print("documents retrieved successfully")
        else:
            print("No documents retrieved for the query.")
    except Exception as e:
        print("Error occurred while retrieving documents:", e)

    # Generate the answer
    completion = palm.generate_text(
        model='models/text-bison-001',
        prompt=query,
        temperature=0.1
    )
    
    return completion.result
