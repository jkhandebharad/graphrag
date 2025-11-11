# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing the CosmosDB vector store implementation."""

import json
import math
from typing import Any

from azure.cosmos import ContainerProxy, CosmosClient, DatabaseProxy
from azure.cosmos.exceptions import CosmosHttpResponseError
from azure.cosmos.partition_key import PartitionKey
from azure.identity import DefaultAzureCredential

from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.data_model.types import TextEmbedder
from graphrag.vector_stores.base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)


class CosmosDBVectorStore(BaseVectorStore):
    """Azure CosmosDB vector storage implementation."""

    _cosmos_client: CosmosClient
    _database_client: DatabaseProxy
    _container_client: ContainerProxy

    def __init__(
        self, vector_store_schema_config: VectorStoreSchemaConfig, **kwargs: Any
    ) -> None:
        super().__init__(
            vector_store_schema_config=vector_store_schema_config, **kwargs
        )

    def connect(self, **kwargs: Any) -> Any:
        """Connect to CosmosDB vector storage."""
        connection_string = kwargs.get("connection_string")
        if connection_string:
            self._cosmos_client = CosmosClient.from_connection_string(connection_string)
        else:
            url = kwargs.get("url")
            if not url:
                msg = "Either connection_string or url must be provided."
                raise ValueError(msg)
            self._cosmos_client = CosmosClient(
                url=url, credential=DefaultAzureCredential()
            )

        database_name = kwargs.get("database_name")
        if database_name is None:
            msg = "Database name must be provided."
            raise ValueError(msg)
        self._database_name = database_name
        if self.index_name is None:
            msg = "Index name is empty or not provided."
            raise ValueError(msg)
        self._container_name = self.index_name

        self.vector_size = self.vector_size
        self._create_database()
        self._create_container()

    def _create_database(self) -> None:
        """Create the database if it doesn't exist."""
        self._cosmos_client.create_database_if_not_exists(id=self._database_name)
        self._database_client = self._cosmos_client.get_database_client(
            self._database_name
        )

    def _delete_database(self) -> None:
        """Delete the database if it exists."""
        if self._database_exists():
            self._cosmos_client.delete_database(self._database_name)

    def _database_exists(self) -> bool:
        """Check if the database exists."""
        existing_database_names = [
            database["id"] for database in self._cosmos_client.list_databases()
        ]
        return self._database_name in existing_database_names

    def _create_container(self) -> None:
        """Create the container if it doesn't exist."""
        partition_key = PartitionKey(path=f"/{self.id_field}", kind="Hash")

        # Define the container vector policy
        vector_embedding_policy = {
            "vectorEmbeddings": [
                {
                    "path": f"/{self.vector_field}",
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": self.vector_size,
                }
            ]
        }

        # Define the vector indexing policy
        indexing_policy = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [
                {"path": "/_etag/?"},
                {"path": f"/{self.vector_field}/*"},
            ],
        }

        # Currently, the CosmosDB emulator does not support the diskANN policy.
        try:
            # First try with the standard diskANN policy
            indexing_policy["vectorIndexes"] = [
                {"path": f"/{self.vector_field}", "type": "diskANN"}
            ]

            # Create the container and container client
            self._database_client.create_container_if_not_exists(
                id=self._container_name,
                partition_key=partition_key,
                indexing_policy=indexing_policy,
                vector_embedding_policy=vector_embedding_policy,
            )
        except CosmosHttpResponseError:
            # If diskANN fails (likely in emulator), retry without vector indexes
            indexing_policy.pop("vectorIndexes", None)

            # Create the container with compatible indexing policy
            self._database_client.create_container_if_not_exists(
                id=self._container_name,
                partition_key=partition_key,
                indexing_policy=indexing_policy,
                vector_embedding_policy=vector_embedding_policy,
            )

        self._container_client = self._database_client.get_container_client(
            self._container_name
        )

    def _delete_container(self) -> None:
        """Delete the vector store container in the database if it exists."""
        if self._container_exists():
            self._database_client.delete_container(self._container_name)

    def _container_exists(self) -> bool:
        """Check if the container name exists in the database."""
        existing_container_names = [
            container["id"] for container in self._database_client.list_containers()
        ]
        return self._container_name in existing_container_names

    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = True
    ) -> None:
        """Load documents into CosmosDB."""
        # Create a CosmosDB container on overwrite
        if overwrite:
            self._delete_container()
            self._create_container()

        if self._container_client is None:
            msg = "Container client is not initialized."
            raise ValueError(msg)

        # Cosmos DB 2MB limit per item (in bytes)
        MAX_DOCUMENT_SIZE = 2 * 1024 * 1024  # 2MB

        # Upload documents to CosmosDB
        for doc in documents:
            if doc.vector is not None:
                print("Document to store:")  # noqa: T201
                print(doc)  # noqa: T201
                
                # Create base document structure to calculate metadata size (without text)
                base_doc_json = {
                    self.id_field: doc.id,
                    self.vector_field: doc.vector,
                    self.text_field: "",  # Empty text for size calculation
                    self.attributes_field: json.dumps(doc.attributes),
                }
                
                # Calculate metadata size (vector + attributes + id, without text)
                metadata_json = json.dumps(base_doc_json, ensure_ascii=False)
                metadata_size = len(metadata_json.encode('utf-8'))
                
                # Calculate available space for text (with 1KB safety margin)
                available_text_size = MAX_DOCUMENT_SIZE - metadata_size - 1024
                
                if available_text_size <= 0:
                    raise ValueError(f"Document {doc.id} metadata alone exceeds 2MB limit (metadata size: {metadata_size} bytes)")
                
                # Get text size in bytes (UTF-8 encoding)
                text = doc.text or ""
                text_bytes = len(text.encode('utf-8')) if text else 0
                
                # Create full document to check total size
                full_doc_json = {
                    self.id_field: doc.id,
                    self.vector_field: doc.vector,
                    self.text_field: text,
                    self.attributes_field: json.dumps(doc.attributes),
                }
                full_doc_size = len(json.dumps(full_doc_json, ensure_ascii=False).encode('utf-8'))
                
                # Check if document fits in one item
                if full_doc_size <= MAX_DOCUMENT_SIZE:
                    # Document fits - store as-is
                    print("Storing document in CosmosDB:")  # noqa: T201
                    print(full_doc_json)  # noqa: T201
                    try:
                        self._container_client.upsert_item(full_doc_json)
                    except CosmosHttpResponseError as e:
                        # If it fails due to size, chunk the document
                        if e.status_code == 413 or e.status_code == 400:
                            print(f"[WARNING] Document {doc.id} exceeded size limit during upsert, attempting chunking...")  # noqa: T201
                            # Continue to chunking logic below
                        else:
                            raise
                else:
                    # Document exceeds 2MB - need to chunk
                    print(f"[INFO] Document {doc.id} exceeds 2MB ({full_doc_size} bytes), splitting into chunks...")  # noqa: T201
                
                # Chunking logic (if document exceeds 2MB or upsert failed)
                if full_doc_size > MAX_DOCUMENT_SIZE or (text_bytes > available_text_size):
                    # Calculate number of chunks needed
                    num_chunks = math.ceil(text_bytes / available_text_size) if text_bytes > 0 else 1
                    
                    print(f"[INFO] Splitting document {doc.id} into {num_chunks} parts...")  # noqa: T201
                    
                    # Prepare chunked attributes
                    chunked_attributes = doc.attributes.copy() if doc.attributes else {}
                    chunked_attributes["_original_id"] = str(doc.id)
                    chunked_attributes["_total_parts"] = num_chunks
                    
                    # Split text into chunks
                    text_bytes_list = text.encode('utf-8')
                    
                    for part_num in range(1, num_chunks + 1):
                        # Calculate chunk boundaries (UTF-8 safe)
                        start_byte = (part_num - 1) * available_text_size
                        end_byte = min(part_num * available_text_size, text_bytes)
                        
                        # Extract chunk bytes
                        chunk_bytes = text_bytes_list[start_byte:end_byte]
                        
                        # Decode chunk with UTF-8 error handling
                        try:
                            chunk_text = chunk_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            # Find the last complete character boundary
                            while len(chunk_bytes) > 0:
                                try:
                                    chunk_text = chunk_bytes.decode('utf-8')
                                    break
                                except UnicodeDecodeError:
                                    chunk_bytes = chunk_bytes[:-1]
                            else:
                                chunk_text = chunk_bytes.decode('utf-8', errors='replace')
                        
                        # Create part ID
                        part_id = f"{doc.id}_part{part_num}"
                        
                        # Update attributes with part number
                        part_attributes = chunked_attributes.copy()
                        part_attributes["_part_number"] = part_num
                        
                        # Create document for this part (same vector, chunked text)
                        part_doc_json = {
                            self.id_field: part_id,
                            self.vector_field: doc.vector,  # Same vector for all parts
                            self.text_field: chunk_text,
                            self.attributes_field: json.dumps(part_attributes),
                        }
                        
                        # Verify part size
                        part_json_str = json.dumps(part_doc_json, ensure_ascii=False)
                        part_size = len(part_json_str.encode('utf-8'))
                        
                        if part_size > MAX_DOCUMENT_SIZE:
                            raise ValueError(
                                f"Document part {part_id} still exceeds 2MB ({part_size} bytes) after chunking. "
                                f"Metadata may be too large (metadata size: {metadata_size} bytes)."
                            )
                        
                        print(f"Storing document part {part_num}/{num_chunks} in CosmosDB: {part_id} ({len(chunk_text)} chars, {part_size} bytes total)")  # noqa: T201
                        try:
                            self._container_client.upsert_item(part_doc_json)
                        except Exception as e:
                            print(f"[ERROR] Failed to store part {part_num}: {e}")  # noqa: T201
                            raise

    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a vector-based similarity search."""
        if self._container_client is None:
            msg = "Container client is not initialized."
            raise ValueError(msg)

        try:
            # Fetch more items to account for chunked documents (k*3 to ensure we get enough unique documents)
            query = f"SELECT TOP {k * 3} c.{self.id_field}, c.{self.text_field}, c.{self.vector_field}, c.{self.attributes_field}, VectorDistance(c.{self.vector_field}, @embedding) AS SimilarityScore FROM c ORDER BY VectorDistance(c.{self.vector_field}, @embedding)"  # noqa: S608
            query_params = [{"name": "@embedding", "value": query_embedding}]
            items = list(
                self._container_client.query_items(
                    query=query,
                    parameters=query_params,
                    enable_cross_partition_query=True,
                )
            )
        except (CosmosHttpResponseError, ValueError):
            # Currently, the CosmosDB emulator does not support the VectorDistance function.
            # For emulator or test environments - fetch all items and calculate distance locally
            query = f"SELECT c.{self.id_field}, c.{self.text_field}, c.{self.vector_field}, c.{self.attributes_field} FROM c"  # noqa: S608
            items = list(
                self._container_client.query_items(
                    query=query,
                    enable_cross_partition_query=True,
                )
            )

            # Calculate cosine similarity locally (1 - cosine distance)
            from numpy import dot
            from numpy.linalg import norm

            def cosine_similarity(a, b):
                if norm(a) * norm(b) == 0:
                    return 0.0
                return dot(a, b) / (norm(a) * norm(b))

            # Calculate scores for all items
            for item in items:
                item_vector = item.get(self.vector_field, [])
                similarity = cosine_similarity(query_embedding, item_vector)
                item["SimilarityScore"] = similarity

            # Sort by similarity score (higher is better) and take top k*3 to account for chunked docs
            items = sorted(
                items, key=lambda x: x.get("SimilarityScore", 0.0), reverse=True
            )[:k * 3]

        # Reassemble chunked documents
        reassembled_docs = {}
        processed_original_ids = set()
        
        for item in items:
            attributes = json.loads(item.get(self.attributes_field, "{}"))
            original_id = attributes.get("_original_id")
            part_number = attributes.get("_part_number")
            total_parts = attributes.get("_total_parts")
            
            if original_id and part_number:
                # This is a chunked document part
                if original_id not in reassembled_docs:
                    reassembled_docs[original_id] = {
                        "parts": {},
                        "vector": item.get(self.vector_field, []),
                        "attributes": attributes.copy(),
                        "score": item.get("SimilarityScore", 0.0),
                        "total_parts": total_parts,
                    }
                    # Remove chunking metadata from attributes
                    reassembled_docs[original_id]["attributes"].pop("_original_id", None)
                    reassembled_docs[original_id]["attributes"].pop("_part_number", None)
                    reassembled_docs[original_id]["attributes"].pop("_total_parts", None)
                
                # Store the text chunk
                reassembled_docs[original_id]["parts"][part_number] = item.get(self.text_field, "")
            else:
                # Regular (non-chunked) document
                doc_id = item.get(self.id_field, "")
                if doc_id not in reassembled_docs:
                    reassembled_docs[doc_id] = {
                        "text": item.get(self.text_field, ""),
                        "vector": item.get(self.vector_field, []),
                        "attributes": attributes,
                        "score": item.get("SimilarityScore", 0.0),
                    }
        
        # Build final results list
        results = []
        for doc_id, doc_data in reassembled_docs.items():
            # Skip if we've already processed this original_id
            if doc_id in processed_original_ids:
                continue
            
            # Check if this is a reassembled chunked document
            if "parts" in doc_data:
                # Reassemble text from parts
                total_parts = doc_data.get("total_parts", len(doc_data["parts"]))
                text_parts = []
                for part_num in range(1, total_parts + 1):
                    if part_num in doc_data["parts"]:
                        text_parts.append(doc_data["parts"][part_num])
                    else:
                        # Missing part - log warning but continue
                        print(f"Warning: Missing part {part_num} for document {doc_id}")  # noqa: T201
                
                reassembled_text = "".join(text_parts)
                
                # Get original_id from the first part's attributes (stored in reassembled_docs)
                original_id = doc_id  # doc_id is the original_id for chunked docs
                
                results.append(
                    VectorStoreSearchResult(
                        document=VectorStoreDocument(
                            id=original_id,
                            text=reassembled_text,
                            vector=doc_data["vector"],
                            attributes=doc_data["attributes"],
                        ),
                        score=doc_data["score"],
                    )
                )
                processed_original_ids.add(original_id)
            else:
                # Regular document
                results.append(
                    VectorStoreSearchResult(
                        document=VectorStoreDocument(
                            id=doc_id,
                            text=doc_data["text"],
                            vector=doc_data["vector"],
                            attributes=doc_data["attributes"],
                        ),
                        score=doc_data["score"],
                    )
                )
                processed_original_ids.add(doc_id)
        
        # Sort by score and return top k (deduplicated)
        results = sorted(results, key=lambda x: x.score, reverse=True)[:k]
        
        return results

    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a text-based similarity search."""
        query_embedding = text_embedder(text)
        if query_embedding:
            return self.similarity_search_by_vector(
                query_embedding=query_embedding, k=k
            )
        return []

    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """Build a query filter to filter documents by a list of ids."""
        if include_ids is None or len(include_ids) == 0:
            self.query_filter = None
        else:
            if isinstance(include_ids[0], str):
                id_filter = ", ".join([f"'{id}'" for id in include_ids])
            else:
                id_filter = ", ".join([str(id) for id in include_ids])
            self.query_filter = (
                f"SELECT * FROM c WHERE c.{self.id_field} IN ({id_filter})"  # noqa: S608
            )
        return self.query_filter

    def search_by_id(self, id: str) -> VectorStoreDocument:
        """Search for a document by id."""
        if self._container_client is None:
            msg = "Container client is not initialized."
            raise ValueError(msg)

        item = self._container_client.read_item(item=id, partition_key=id)
        return VectorStoreDocument(
            id=item.get(self.id_field, ""),
            vector=item.get(self.vector_field, []),
            text=item.get(self.text_field, ""),
            attributes=(json.loads(item.get(self.attributes_field, "{}"))),
        )

    def clear(self) -> None:
        """Clear the vector store."""
        self._delete_container()
        self._delete_database()
