# Optimized RAG System Code with Pinecone Namespaces and Mistral LLM

# This code replaces Gemini with a generic Mistral API endpoint for LLM interactions.

import os
import uuid
import requests
import io
import time
import asyncio
import json
import re 
from typing import List, Optional

# --- Imports from External Libraries ---
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, HttpUrl
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai # Keep for embedding, but remove for LLM calls
from pydantic_settings import BaseSettings
from pypdf import PdfReader
import httpx 

# --- 1. APPLICATION SETUP & CONFIGURATION ---
load_dotenv() 

class Settings(BaseSettings):
    PINECONE_API_KEY: str
    GOOGLE_API_KEY: str # Still needed for Google embeddings
    MISTRAL_API_KEY: str # New: API key for Mistral
    MISTRAL_API_URL: str # New: API URL for Mistral (no default)
    PINECONE_SINGLE_INDEX_NAME: str = "hackrx-rag-data" 
    class Config:
        env_file = ".env"

settings = Settings()

app = FastAPI(
    title="HackRx 6.0 Winning Submission (Optimized with Namespaces & Mistral)",
    description="An advanced RAG API with Multi-Query Retrieval for maximum accuracy, optimized for performance and using Pinecone namespaces, with Mistral LLM.",
)

# Configure services
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
genai.configure(api_key=settings.GOOGLE_API_KEY) # Keep for Google embeddings

# --- Mistral LLM Configuration ---
# This is a placeholder for a generic Mistral API call. 
# You might need to adjust based on the specific Mistral API provider (e.g., mistral.ai, Together AI, Anyscale)
async def call_mistral_api(prompt: str, temperature: float = 0.2) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.MISTRAL_API_KEY}"
    }
    payload = {
        "model": "mistral-small-latest", # Changed from mixtral-8x7b-instruct-v0.1 to a valid API model
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 512 # Reduced for faster inference
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(settings.MISTRAL_API_URL, headers=headers, json=payload, timeout=60.0)
        try:
            response.raise_for_status() # Raise an exception for HTTP errors
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            raise # Re-raise the exception after printing details
        return response.json()["choices"][0]["message"]["content"]

EMBEDDING_MODEL_NAME = "models/text-embedding-004"
EMBEDDING_DIMENSION = 768

# In-memory cache for document_url to namespace mapping
DOCUMENT_CACHE = {}

PINECONE_EFFECTIVE_CHAR_LIMIT = 30000 

# --- 2. PYDANTIC MODELS ---
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Utility for Retries ---
async def retry_async(func, retries=3, delay=1, backoff=2, exceptions=(Exception, httpx.HTTPStatusError)):
    attempt = 0
    while attempt < retries:
        try:
            return await func()
        except exceptions as e:
            print(f"Attempt {attempt + 1} failed: {type(e).__name__} - {e}")
            if isinstance(e, httpx.HTTPStatusError):
                print(f"HTTP Error Details: Status Code {e.response.status_code}, Response Text: {e.response.text}")
            attempt += 1
            if attempt < retries:
                await asyncio.sleep(delay)
                delay *= backoff
    raise 

# --- 3. RAG PIPELINE LOGIC ---

def get_text_chunks(text: str) -> List[str]:
    """Improved, more robust chunking strategy without external libraries.
    This version aims to preserve semantic meaning better by prioritizing sentence boundaries
    and handling short paragraphs more gracefully.
    """
    final_chunks = []
    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:  
            continue

        sentences = re.split(r'(?<=[.!?])\s+', paragraph) 
        
        current_chunk_sentences = []
        current_chunk_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: continue

            if not re.search(r'[.!?]$', sentence):
                sentence_to_add = sentence + ". " 
            else:
                sentence_to_add = sentence + " " 
            
            if (current_chunk_len + len(sentence_to_add) < PINECONE_EFFECTIVE_CHAR_LIMIT and 
                len(("".join(current_chunk_sentences) + sentence_to_add).encode("utf-8")) < 40000):
                current_chunk_sentences.append(sentence_to_add)
                current_chunk_len += len(sentence_to_add)
            else:
                if current_chunk_sentences:
                    final_chunks.append("".join(current_chunk_sentences).strip())
                
                current_chunk_sentences = [sentence_to_add]
                current_chunk_len = len(sentence_to_add)
        
        if current_chunk_sentences:
            final_chunks.append("".join(current_chunk_sentences).strip())
                
    merged_chunks = []
    if final_chunks:
        merged_chunks.append(final_chunks[0])
        for i in range(1, len(final_chunks)):
            if len(final_chunks[i]) < 100 and \
               len((merged_chunks[-1] + " " + final_chunks[i]).encode("utf-8")) < 40000 and \
               len(merged_chunks[-1] + " " + final_chunks[i]) < PINECONE_EFFECTIVE_CHAR_LIMIT:
                merged_chunks[-1] += " " + final_chunks[i]
            else:
                merged_chunks.append(final_chunks[i])
    
    return merged_chunks



async def get_mistral_response_async(question: str, context: str) -> str:
    """Asynchronously generates an answer from Mistral with more granular error reporting and retries."""
    prompt = f"You are an expert AI assistant specializing in analyzing policy documents. Your task is to answer the user\'s question based ONLY on the provided context text.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:\nProvide a direct and concise answer based on the context. If the answer is not contained within the context, state \'The answer could not be found in the provided text.\'"
    try:
        response_content = await retry_async(lambda: call_mistral_api(prompt, temperature=0.2))
        answer_text = response_content.strip()
        if "The answer could not be found in the provided text." in answer_text:
            return "Answer not found in provided context."
        return answer_text
    except Exception as e:
        print(f"Error generating Mistral response: {e}")
        return "LLM API Error: Could not generate an answer due to an internal issue."

async def setup_and_process_document(index, doc_url: str, namespace: str):
    """Downloads, chunks, embeds, and upserts a document into a persistent index using a namespace."""
    print(f"Processing new document from: {doc_url} into namespace: {namespace}")
    
    response = requests.get(doc_url)
    response.raise_for_status() 
    pdf_file = io.BytesIO(response.content)
    reader = PdfReader(pdf_file)
    document_text = "".join(page.extract_text() for page in reader.pages)
    text_chunks = get_text_chunks(document_text) 
    print(f"Split document into {len(text_chunks)} chunks.")
    
    print("Generating embeddings with Google\'s async API...")
    embedding_result = await genai.embed_content_async(model=EMBEDDING_MODEL_NAME, content=text_chunks, task_type="RETRIEVAL_DOCUMENT")
    embeddings = embedding_result["embedding"]
    
    vectors_to_upsert = [
        {"id": f"chunk_{i}", "values": embedding, "metadata": {"text": chunk}}
        for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings))
    ]
    
    for i in range(0, len(vectors_to_upsert), 100):
        batch = vectors_to_upsert[i:i+100]
        index.upsert(vectors=batch, namespace=namespace) 
    print(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone in namespace {namespace}.")

async def get_single_answer(index, question: str, namespace: str) -> str:
    """Processes a single question using a direct retrieval strategy with namespace filtering."""
    embedding_result = await genai.embed_content_async(model=EMBEDDING_MODEL_NAME, content=question, task_type="RETRIEVAL_QUERY")
    query_embedding = embedding_result["embedding"]
    
    query_results = index.query(
        vector=query_embedding,
        top_k=5, # Reduced top_k for faster retrieval
        include_metadata=True,
        namespace=namespace 
    )
    
    retrieved_chunks = {}
    for match in query_results["matches"]:
        if match["metadata"]["text"] not in retrieved_chunks:
            retrieved_chunks[match["metadata"]["text"]] = match["score"]
                
    context = " ".join(retrieved_chunks.keys())
    print(f"\n--- CONTEXT for Q: \'{question}\' ---\n{context}\n-----------------------------------\n")
    
    answer_text = await get_mistral_response_async(question, context) # Using Mistral for answer generation
    print(f"Q: {question}\nA: {answer_text}")
    return answer_text

async def answer_questions_from_index(index, questions: List[str], namespace: str) -> List[str]:
    """Answers all questions concurrently."""
    print("Answering all questions concurrently using Multi-Query...")
    tasks = [get_single_answer(index, q, namespace) for q in questions]
    final_answers = await asyncio.gather(*tasks)
    return final_answers

# --- 4. API ENDPOINTS ---
@app.post("/hackrx/run", response_model=HackRxResponse, tags=["RAG System"])
async def run_submission(request: HackRxRequest, authorization: Optional[str] = Header(None)):
    doc_url = str(request.documents)
    
    index_name = settings.PINECONE_SINGLE_INDEX_NAME
    
    try:
        index = pc.Index(index_name)

        namespace = DOCUMENT_CACHE.get(doc_url)
        if namespace:
            print(f"CACHE HIT: Reusing namespace \'{namespace}\' for document: {doc_url}")
        else:
            namespace = uuid.uuid5(uuid.NAMESPACE_URL, doc_url).hex 
            print(f"CACHE MISS: Processing new document: {doc_url} with new namespace: {namespace}")
            await setup_and_process_document(index, doc_url, namespace)
            DOCUMENT_CACHE[doc_url] = namespace

        answers = await answer_questions_from_index(index, request.questions, namespace)
        return HackRxResponse(answers=answers)
    except Exception as e:
        print(f"An error occurred: {e}")
        if "Forbidden" in str(e) or "max serverless indexes allowed" in str(e):
            raise HTTPException(status_code=403, detail="Pinecone Error: Max serverless indexes reached. Please use an existing index or upgrade your plan.")
        elif "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Pinecone Error: Index \'{index_name}\' not found. Please ensure the index exists in your Pinecone account.")
        else:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "HackRx API is running!"}


