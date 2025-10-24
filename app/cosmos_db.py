"""
CosmosDB-based Document Metadata Management
Replaces SQLite with CosmosDB for document tracking
"""
from typing import List, Optional
from app.model import Document
from app.cosmos_input import InputManager


class CosmosDocumentDB:
    """Document metadata management using CosmosDB."""
    
    def __init__(self):
        self.input_manager = InputManager()
    
    def init_db(self):
        """Initialize database (no-op for CosmosDB as containers are auto-created)."""
        pass
    
    def insert_document(self, doc: Document):
        """
        Insert document metadata into CosmosDB.
        Note: The actual file content should be stored separately using InputManager.
        """
        # This is mainly for compatibility - actual storage happens in InputManager
        pass
    
    def get_unindexed_documents(self, case_id: str) -> List[Document]:
        """Get all unindexed documents for a case."""
        cosmos_docs = self.input_manager.list_unindexed_documents(case_id)
        
        documents = []
        for cosmos_doc in cosmos_docs:
            # Convert CosmosDB document to Document model
            doc = Document(
                firm_id=cosmos_doc.get("firm_id", "1"),
                case_id=cosmos_doc["case_id"],
                document_id=cosmos_doc["document_id"],
                filename=cosmos_doc["filename"],
                file_path=f"cosmos://{case_id}/{cosmos_doc['document_id']}",  # Virtual path
                is_indexed=cosmos_doc.get("is_indexed", False)
            )
            documents.append(doc)
        
        return documents
    
    def get_next_document_id(self, case_id: str) -> int:
        """Get the next available document ID for a case."""
        return self.input_manager.get_next_document_id(case_id)
    
    def mark_as_indexed(self, case_id: str, document_id: int):
        """Mark a document as indexed."""
        self.input_manager.mark_as_indexed(case_id, document_id)


# Global instance for compatibility
_cosmos_db: Optional[CosmosDocumentDB] = None

def get_cosmos_db() -> CosmosDocumentDB:
    """Get or create the global CosmosDB document database instance."""
    global _cosmos_db
    if _cosmos_db is None:
        _cosmos_db = CosmosDocumentDB()
    return _cosmos_db

def init_db():
    """Initialize CosmosDB (compatibility function)."""
    get_cosmos_db().init_db()

def insert_document(doc: Document):
    """Insert a document (compatibility function)."""
    get_cosmos_db().insert_document(doc)

def get_unindexed_documents(case_id: str) -> List[Document]:
    """Get unindexed documents (compatibility function)."""
    return get_cosmos_db().get_unindexed_documents(case_id)

def get_next_document_id(case_id: str) -> int:
    """Get next document ID (compatibility function)."""
    return get_cosmos_db().get_next_document_id(case_id)

def mark_as_indexed(case_id: str, document_id: int):
    """Mark as indexed (compatibility function)."""
    get_cosmos_db().mark_as_indexed(case_id, document_id)
