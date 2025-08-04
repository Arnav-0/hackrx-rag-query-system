# embedder.py

import os
import openai
import pinecone
from uuid import uuid4
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Load OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embedding(text: str) -> List[float]:
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response["data"][0]["embedding"]

def embed_and_store(chunks: List[str], namespace: str):
    vectors = []
    for chunk in chunks:
        vector = get_embedding(chunk)
        uid = str(uuid4())
        vectors.append((uid, vector, {"text": chunk}))
    
    index.upsert(vectors=vectors, namespace=namespace)
