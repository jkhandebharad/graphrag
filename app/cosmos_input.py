"""
CosmosDB Input Container Manager
Handles storage and retrieval of input documents by case_id
"""
import os
import base64
from typing import Optional, List, Dict, Any
from datetime import datetime
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from app.cosmos_client import get_cosmos_manager
import logging


class InputManager:
    """Manages input documents in CosmosDB for GraphRAG indexing."""

    def __init__(self):
        self.manager = None  # Will be set by get_firm_managers()
        self.container = None  # Will be set by get_firm_managers()
    
    def _normalize_caseid(self, case_id: str) -> str:
        """Normalize case_id for partition key (numeric cases get .txt extension)."""
        return f"{case_id}.txt" if case_id.isdigit() else case_id

    # ==========================================================
    #  1️⃣ Store Binary Document (e.g., PDF before OCR)
    # ==========================================================
    def store_document(
        self,
        case_id: str,
        document_id: int,
        filename: str,
        content: bytes,
        firm_id: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Store a binary document in the input container."""
        case_id = str(case_id)
        content_base64 = base64.b64encode(content).decode("utf-8")

        doc = {
            "id": f"input-{case_id}-{document_id}",
            "caseid": self._normalize_caseid(case_id),  # Partition key - normalized with .txt for numeric cases
            "case_id": case_id,   # Original case_id for queries/metadata
            "firm_id": firm_id,
            "document_id": document_id,
            "filename": filename,
            "content": content_base64,
            "content_type": content_type,
            "is_indexed": False,
            "is_text": "false",
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        return self.container.upsert_item(doc)

    # ==========================================================
    #  2️⃣ Store Extracted Text (used by GraphRAG input)
    # ==========================================================
    def store_extracted_text(
        self,
        case_id: str,
        document_id: int,
        text_content: str,
        filename: str,
        original_filename: str = None,
        firm_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store extracted plain text for GraphRAG to read directly from CosmosDB.
        Uses individual document structure (not nested).
        """
        case_id = str(case_id)

        # Sanitize filename for CosmosDB (keep .txt extension as-is)
        safe_filename = (
            filename.replace("/", "-")
            .replace(":", "-")
            .replace("#", "-")
            .replace("?", "-")
            .replace("\\", "-")
        )

        # Store individual document (GraphRAG will read directly from this)
        doc = {
            "id": safe_filename,  # Use filename as document ID
            "caseid": safe_filename,  # Use id as partition key for GraphRAG compatibility
            "case_id": case_id,   # Original case_id for queries/metadata
            "document_id": document_id,
            "filename": filename,
            "original_filename": original_filename or filename,
            "firm_id": firm_id,
            "content": text_content,
            "content_type": "text/plain",
            "is_text": "true",      # Must be string "true" for file_filter
            "is_indexed": False,    # Helps incremental indexing
            "created_at": datetime.utcnow().isoformat(),
        }

        return self.container.upsert_item(doc)

    # ==========================================================
    #  3️⃣ Retrieve Single Document
    # ==========================================================
    def get_document(self, case_id: str, document_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a document by case_id and document_id."""
        case_id = str(case_id)
        
        # Get firm_id from environment for filename format
        firm_id = os.getenv("FIRM_ID", "1")
        filename = f"{firm_id}_{case_id}_{document_id}.txt"
        
        try:
            # Try to read document directly using filename as partition key
            doc = self.container.read_item(item=filename, partition_key=filename)
            return doc
        except Exception:
            # Fallback to query if direct read fails
            query = """
            SELECT * FROM c 
            WHERE c.case_id = @case_id 
            AND c.document_id = @document_id
            """
            items = list(
                self.container.query_items(
                    query=query,
                    parameters=[
                        {"name": "@case_id", "value": case_id},
                        {"name": "@document_id", "value": document_id},
                    ],
                    enable_cross_partition_query=True,
                )
            )
            return items[0] if items else None

    # ==========================================================
    #  4️⃣ Get Document Content (text)
    # ==========================================================
    def get_document_content(self, case_id: str, document_id: int) -> Optional[str]:
        """Return text content for a document."""
        doc = self.get_document(case_id, document_id)
        if doc and "content" in doc:
            return doc["content"]
        return None

    # ==========================================================
    #  5️⃣ List All or Text-Only Documents
    # ==========================================================
    def list_documents(self, case_id: str, text_only: bool = False) -> List[Dict[str, Any]]:
        """List all documents (or text-only ones) for a case."""
        case_id = str(case_id)
        query = "SELECT * FROM c WHERE c.case_id = @case_id"
        if text_only:
            query += " AND c.is_text = 'true'"

        items = self.container.query_items(
            query=query,
            parameters=[{"name": "@case_id", "value": case_id}],
            enable_cross_partition_query=True,  # Enable cross-partition since we're not using case_id as partition key
        )
        return list(items)

    # ==========================================================
    #  6️⃣ List Unindexed Texts (for incremental indexing)
    # ==========================================================
    def list_unindexed_documents(self, case_id: str) -> List[Dict[str, Any]]:
        """List unindexed text documents for a case."""
        case_id = str(case_id)
        query = """
        SELECT * FROM c 
        WHERE c.case_id = @case_id 
        AND c.is_text = 'true'
        AND (c.is_indexed = false OR NOT IS_DEFINED(c.is_indexed))
        """
        items = self.container.query_items(
            query=query,
            parameters=[{"name": "@case_id", "value": case_id}],
            enable_cross_partition_query=True,  # Enable cross-partition since we're not using case_id as partition key
        )
        return list(items)

    # ==========================================================
    #  7️⃣ Mark Document as Indexed
    # ==========================================================
    def mark_as_indexed(self, case_id: str, document_id: int):
        """Set is_indexed=True after GraphRAG completes indexing."""
        case_id = str(case_id)
        
        # Get firm_id from environment for filename format
        firm_id = os.getenv("FIRM_ID", "1")
        filename = f"{firm_id}_{case_id}_{document_id}.txt"
        
        try:
            # Read document directly using filename as partition key
            doc = self.container.read_item(item=filename, partition_key=filename)
            doc["is_indexed"] = True
            doc["indexed_at"] = datetime.utcnow().isoformat()
            self.container.upsert_item(doc)
        except Exception:
            print(
                f"[WARNING] Text document not found for marking as indexed: case_id={case_id}, document_id={document_id}"
            )

    # ==========================================================
    #  8️⃣ Next Document ID
    # ==========================================================
    def get_next_document_id(self, case_id: str) -> int:
        """Return next available numeric document_id for a case."""
        case_id = str(case_id)
        query = "SELECT VALUE MAX(c.document_id) FROM c WHERE c.case_id = @case_id"
        items = list(
            self.container.query_items(
                query=query,
                parameters=[{"name": "@case_id", "value": case_id}],
                enable_cross_partition_query=True,  # Enable cross-partition since we're not using case_id as partition key
            )
        )
        max_id = items[0] if items and items[0] is not None else 0
        return int(max_id) + 1

    # ==========================================================
    #  9️⃣ Delete Document (binary + text)
    # ==========================================================
    def delete_document(self, case_id: str, document_id: int):
        """Delete a document and its text version from CosmosDB."""
        case_id = str(case_id)
        
        # Get firm_id from environment for filename format
        firm_id = os.getenv("FIRM_ID", "1")
        
        # Try to delete text record with .txt extension (current format: {firm_id}_{case_id}_{doc_id}.txt)
        try:
            txt_filename = f"{firm_id}_{case_id}_{document_id}.txt"
            self.container.delete_item(
                item=txt_filename, partition_key=txt_filename  # Use id as partition key
            )
        except CosmosResourceNotFoundError:
            pass
        
        # Try to delete old format (.json) if it exists for backward compatibility
        try:
            json_id = f"{document_id}.json"
            self.container.delete_item(
                item=json_id, partition_key=json_id  # Use id as partition key
            )
        except CosmosResourceNotFoundError:
            pass
        
        # Try to delete very old format (input-{case_id}-{doc_id}) if it exists
        try:
            old_id = f"input-{case_id}-{document_id}"
            self.container.delete_item(
                item=old_id, partition_key=old_id  # Use id as partition key
            )
        except CosmosResourceNotFoundError:
            pass
