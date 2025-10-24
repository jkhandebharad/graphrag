"""
Document Database Module
Now uses CosmosDB instead of SQLite
"""
from app.cosmos_db import (
    init_db,
    insert_document,
    get_unindexed_documents,
    get_next_document_id,
    mark_as_indexed
)

# Re-export all functions for backward compatibility
__all__ = [
    'init_db',
    'insert_document',
    'get_unindexed_documents',
    'get_next_document_id',
    'mark_as_indexed'
]
