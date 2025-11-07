"""
CosmosDB Input Container Manager
Handles storage and retrieval of input documents by case_id
"""
import os
import base64
import json
import math
from typing import Optional, List, Dict, Any
from datetime import datetime
from azure.cosmos.exceptions import CosmosResourceNotFoundError, CosmosHttpResponseError
from app.cosmos_client import get_cosmos_manager
import logging


class InputManager:
    """Manages input documents in CosmosDB for GraphRAG indexing."""

    def __init__(self):
        self.manager = None  # Will be set by get_firm_managers()
        self.container = None  # Will be set by get_firm_managers()
    
    # No longer needed - using /id as partition key in case-specific containers

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
            "case_id": case_id,   # Keep for queries/metadata
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
        If document (metadata + content) exceeds 2MB, splits into multiple parts with _part1, _part2, etc.
        Uses individual document structure (not nested).
        """
        case_id = str(case_id)
        MAX_DOCUMENT_SIZE = 2 * 1024 * 1024  # 2MB in bytes
        
        # Base filename without extension for part naming
        base_filename = filename.replace(".txt", "")
        
        # Sanitize filename for CosmosDB (keep .txt extension as-is)
        safe_filename = (
            filename.replace("/", "-")
            .replace(":", "-")
            .replace("#", "-")
            .replace("?", "-")
            .replace("\\", "-")
        )
        
        # Create a test document to calculate metadata size (NOT saved to DB)
        test_doc = {
            "id": safe_filename,
            "case_id": case_id,
            "document_id": document_id,
            "filename": filename,
            "original_filename": original_filename or filename,
            "firm_id": firm_id or "",
            "content": "",  # Placeholder - only for size calculation
            "content_type": "text/plain",
            "is_text": "true",
            "is_indexed": False,
            "created_at": datetime.utcnow().isoformat(),
        }
        
        # Calculate metadata size (without content) - this test_doc is NOT saved
        metadata_json = json.dumps(test_doc, ensure_ascii=False)
        metadata_size = len(metadata_json.encode('utf-8'))
        
        # Calculate available space for content (with 1KB buffer for safety)
        available_content_size = MAX_DOCUMENT_SIZE - metadata_size - 1024
        
        # If available space is too small, recalculate with part number in filename
        if available_content_size < 1000:  # If less than 1KB available, metadata might be too large
            # Recalculate with part1 filename to get accurate metadata size for chunked documents
            part_filename = f"{base_filename}_part1.txt"
            safe_part_filename = (
                part_filename.replace("/", "-")
                .replace(":", "-")
                .replace("#", "-")
                .replace("?", "-")
                .replace("\\", "-")
            )
            test_doc_part = {
                "id": safe_part_filename,
                "case_id": case_id,
                "document_id": document_id,
                "filename": part_filename,
                "original_filename": original_filename or filename,
                "firm_id": firm_id or "",
                "content": "",
                "content_type": "text/plain",
                "is_text": "true",
                "is_indexed": False,
                "created_at": datetime.utcnow().isoformat(),
                "part_number": 1,
                "total_parts": 1,
            }
            metadata_json_part = json.dumps(test_doc_part, ensure_ascii=False)
            metadata_size = len(metadata_json_part.encode('utf-8'))
            available_content_size = MAX_DOCUMENT_SIZE - metadata_size - 1024
        
        # Get content size in bytes (UTF-8 encoding)
        content_bytes = text_content.encode('utf-8')
        content_size = len(content_bytes)
        
        # Check if single document fits (test with actual content)
        test_doc["content"] = text_content
        full_doc_json = json.dumps(test_doc, ensure_ascii=False)
        full_doc_size = len(full_doc_json.encode('utf-8'))
        
        # If document fits in one piece, store as single document
        if full_doc_size <= MAX_DOCUMENT_SIZE and content_size <= available_content_size:
            doc = {
                "id": safe_filename,
                "case_id": case_id,
                "document_id": document_id,
                "filename": filename,
                "original_filename": original_filename or filename,
                "firm_id": firm_id,
                "content": text_content,
                "content_type": "text/plain",
                "is_text": "true",
                "is_indexed": False,
                "created_at": datetime.utcnow().isoformat(),
            }
            
            try:
                return self.container.upsert_item(doc)
            except CosmosHttpResponseError as e:
                # If it fails due to size, chunk the document
                if e.status_code == 413 or e.status_code == 400:
                    print(f"[WARNING] Document exceeded size limit ({full_doc_size} bytes), attempting chunking...")
                    # Continue to chunking logic below
                else:
                    raise
        
        # Need to chunk the document
        # Calculate number of chunks needed
        num_chunks = math.ceil(content_size / available_content_size)
        
        print(f"[INFO] Document size ({full_doc_size} bytes) exceeds 2MB limit. Splitting into {num_chunks} parts...")
        
        # Store all parts
        stored_parts = []
        for part_num in range(1, num_chunks + 1):
            # Calculate chunk boundaries
            start_idx = (part_num - 1) * available_content_size
            end_idx = min(start_idx + available_content_size, content_size)
            
            # Extract chunk bytes
            chunk_bytes = content_bytes[start_idx:end_idx]
            
            # Decode chunk with UTF-8 error handling
            try:
                chunk_content = chunk_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # Find the last complete character boundary
                while len(chunk_bytes) > 0:
                    try:
                        chunk_content = chunk_bytes.decode('utf-8')
                        break
                    except UnicodeDecodeError:
                        chunk_bytes = chunk_bytes[:-1]
                else:
                    chunk_content = chunk_bytes.decode('utf-8', errors='replace')
            
            # Create part filename
            part_filename = f"{base_filename}_part{part_num}.txt"
            
            # Sanitize part filename
            safe_part_filename = (
                part_filename.replace("/", "-")
                .replace(":", "-")
                .replace("#", "-")
                .replace("?", "-")
                .replace("\\", "-")
            )
            
            # Create document for this part
            doc = {
                "id": safe_part_filename,
                "case_id": case_id,
                "document_id": document_id,
                "filename": part_filename,
                "original_filename": original_filename or filename,
                "firm_id": firm_id,
                "content": chunk_content,
                "content_type": "text/plain",
                "is_text": "true",
                "is_indexed": False,
                "created_at": datetime.utcnow().isoformat(),
                "part_number": part_num,
                "total_parts": num_chunks,
            }
            
            # Verify this part doesn't exceed 2MB
            part_json = json.dumps(doc, ensure_ascii=False)
            part_size = len(part_json.encode('utf-8'))
            
            if part_size > MAX_DOCUMENT_SIZE:
                raise ValueError(f"Part {part_num} still exceeds 2MB limit ({part_size} bytes) after chunking. Metadata may be too large.")
            
            try:
                result = self.container.upsert_item(doc)
                stored_parts.append(result)
                print(f"   [SUCCESS] Stored part {part_num}/{num_chunks}: {part_filename} ({len(chunk_content)} chars, {part_size} bytes total)")
            except Exception as e:
                print(f"   [ERROR] Failed to store part {part_num}: {e}")
                raise
        
        # Return the first part's result (for backward compatibility)
        # The calling code can check for part_number field to know if document was chunked
        if not stored_parts:
            raise RuntimeError("Failed to store any document parts after chunking")
        return stored_parts[0]

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
