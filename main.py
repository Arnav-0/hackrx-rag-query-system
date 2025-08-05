# main.py - The Final, Self-Evaluating Code for Submission

import os
import uuid
import requests
import io
import time
from typing import List, Optional

# --- Imports from External Libraries ---
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, HttpUrl
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from pydantic_settings import BaseSettings
from pypdf import PdfReader

# --- 1. APPLICATION SETUP & CONFIGURATION ---
load_dotenv() 

class Settings(BaseSettings):
    PINECONE_API_KEY: str
    GOOGLE_API_KEY: str
    class Config:
        env_file = ".env"

settings = Settings()

app = FastAPI(
    title="HackRx 6.0 Winning Submission",
    description="A self-evaluating RAG API with performance and confidence metrics.",
)

# Configure services
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
genai.configure(api_key=settings.GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

EMBEDDING_MODEL_NAME = 'models/text-embedding-004'
EMBEDDING_DIMENSION = 768

# In-memory cache
DOCUMENT_CACHE = {}
PINECONE_METADATA_LIMIT = 38000

# --- 2. PYDANTIC MODELS (with Evaluation Metrics) ---

class Answer(BaseModel):
    question: str
    answer: str
    retrieval_score: float
    context: str

class EvaluationMetrics(BaseModel):
    processing_time_seconds: float
    document_source: str
    
class HackRxResponse(BaseModel):
    answers: List[Answer]
    evaluation: EvaluationMetrics

class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

# --- 3. RAG PIPELINE LOGIC ---

def get_text_chunks(text: str) -> List[str]:
    """Robust, two-tiered chunking strategy."""
    final_chunks = []
    paragraphs = text.split('\n\n')
    for paragraph in paragraphs:
        if len(paragraph.encode('utf-8')) < PINECONE_METADATA_LIMIT:
            if len(paragraph.strip()) > 100: 
                final_chunks.append(paragraph.strip())
        else:
            print(f"WARNING: A paragraph of size {len(paragraph.encode('utf-8'))} bytes exceeds the limit. Splitting it.")
            start, chunk_size, chunk_overlap = 0, 8000, 400
            while start < len(paragraph):
                end = start + chunk_size
                final_chunks.append(paragraph[start:end])
                start += chunk_size - chunk_overlap
    return final_chunks

def get_gemini_response(question: str, context: str) -> str:
    """Generates an answer from Gemini based on the provided context."""
    prompt = f"You are an expert AI assistant specializing in analyzing policy documents. Your task is to answer the user's question based ONLY on the provided context text.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:\nProvide a direct and concise answer. If the answer is not in the context, state 'The answer could not be found in the provided text.'"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating Gemini response: {e}")
        return "Error: Could not generate an answer."

def setup_and_process_document(index, doc_url: str):
    """Downloads, chunks, embeds, and upserts a document."""
    print(f"Processing new document from: {doc_url}")
    response = requests.get(doc_url)
    response.raise_for_status() 
    pdf_file = io.BytesIO(response.content)
    reader = PdfReader(pdf_file)
    document_text = "".join(page.extract_text() for page in reader.pages)
    text_chunks = get_text_chunks(document_text)
    print(f"Split document into {len(text_chunks)} chunks.")
    
    embedding_result = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=text_chunks, task_type="RETRIEVAL_DOCUMENT")
    embeddings = embedding_result['embedding']
    
    vectors_to_upsert = [{"id": f"chunk_{i}", "values": embedding, "metadata": {"text": chunk}} for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings))]
    for i in range(0, len(vectors_to_upsert), 100):
        index.upsert(vectors=vectors_to_upsert[i:i+100])
    print(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone.")

def answer_questions_from_index(index, questions: List[str]) -> List[Answer]:
    """Answers all questions sequentially and returns a rich Answer object."""
    print("Waiting for Pinecone to finish indexing...")
    time.sleep(10)
    print("Answering questions sequentially...")
    final_answers = []
    for question in questions:
        question_embedding_result = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=question, task_type="RETRIEVAL_QUERY")
        question_embedding = question_embedding_result['embedding']
        
        query_results = index.query(vector=question_embedding, top_k=5, include_metadata=True)
        
        context = " ".join([match['metadata']['text'] for match in query_results['matches']])
        # Get the confidence score of the single most relevant chunk
        top_score = query_results['matches'][0]['score'] if query_results['matches'] else 0.0
        
        answer_text = get_gemini_response(question, context)
        
        final_answers.append(Answer(
            question=question, 
            answer=answer_text, 
            context=context,
            retrieval_score=top_score
        ))
        print(f"Q: {question}\nA: {answer_text} (Score: {top_score:.4f})")
    return final_answers

# --- 4. API ENDPOINTS ---

@app.post("/hackrx/run", response_model=HackRxResponse, tags=["RAG System"])
def run_submission(request: HackRxRequest, authorization: Optional[str] = Header(None)):
    start_time = time.perf_counter()
    doc_url = str(request.documents)
    index_name = None
    try:
        if doc_url in DOCUMENT_CACHE:
            index_name = DOCUMENT_CACHE[doc_url]
            print(f"CACHE HIT: Reusing Pinecone index '{index_name}' for document: {doc_url}")
            index = pc.Index(index_name)
        else:
            print(f"CACHE MISS: Processing new document: {doc_url}")
            index_name = f"hackrx-session-{uuid.uuid4().hex[:8]}"
            pc.create_index(
                name=index_name, 
                dimension=EMBEDDING_DIMENSION, 
                metric="cosine", 
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            index = pc.Index(index_name)
            setup_and_process_document(index, doc_url)
            DOCUMENT_CACHE[doc_url] = index_name

        answers = answer_questions_from_index(index, request.questions)
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        evaluation_metrics = EvaluationMetrics(
            processing_time_seconds=processing_time,
            document_source=doc_url
        )
        
        return HackRxResponse(answers=answers, evaluation=evaluation_metrics)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    # Note: Caching logic means indexes are not deleted. This is fine for the demo
    # but would need a cleanup strategy in production.

@app.get("/", tags=["Health Check"])
def root():
    return {"message": "HackRx API is running!"}