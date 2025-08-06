# retriever.py

import os
import openai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Load API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

def get_question_embedding(question: str):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=question
    )
    return response["data"][0]["embedding"]

def retrieve_relevant_chunks(question: str, namespace: str, top_k: int = 5):
    query_vector = get_question_embedding(question)
    
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

    chunks = [match["metadata"]["text"] for match in results["matches"]]
    return chunks
