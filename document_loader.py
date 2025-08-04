# document_loader.py

import httpx
import fitz  # PyMuPDF
import docx
import tempfile
import os
from typing import List
from bs4 import BeautifulSoup

async def fetch_and_parse_document(url: str) -> List[str]:
    # Step 1: Download the file
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()

    file_extension = url.split('.')[-1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    # Step 2: Parse based on file type
    if file_extension == "pdf":
        text = extract_text_from_pdf(tmp_path)
    elif file_extension in ["docx", "doc"]:
        text = extract_text_from_docx(tmp_path)
    elif file_extension in ["eml", "txt"]:
        text = extract_text_from_email(tmp_path)
    else:
        text = ""

    os.remove(tmp_path)

    # Step 3: Clean + Chunk
    chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 30]
    return chunks

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_email(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
        soup = BeautifulSoup(raw, 'html.parser')
        return soup.get_text()
