from dataclasses import dataclass
from pydantic import BaseModel, Field

@dataclass
class SearchQuery:
    title: str
    text: str

class Doc(BaseModel):
    doc_id: str = Field(..., description="Unique ID for documents")
    name: str = Field(..., description="document name")
    text: str = Field(..., description="Raw text of document")

class Docevaluation(BaseModel):
    doc_id: str
    summary: str
