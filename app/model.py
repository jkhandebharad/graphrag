from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    """A data model for a single document in our system."""
    firm_id: str
    case_id: str
    document_id: int
    filename: str
    file_path: str
    is_indexed: Optional[bool] = False
