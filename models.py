# models.py

from pydantic import BaseModel, HttpUrl
from typing import List

class HackRxRequest(BaseModel):
    """The structure of the incoming request from the hackathon platform."""
    documents: HttpUrl
    questions: List[str]

class Answer(BaseModel):
    """The structure for a single answer, including its source context."""
    question: str
    answer: str
    context: str

class HackRxResponse(BaseModel):
    """The final response structure sent back by the API."""
    answers: List[Answer]