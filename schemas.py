# schemas.py

from typing import List
from pydantic import BaseModel

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]
