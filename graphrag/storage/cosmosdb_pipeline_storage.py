# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Azure CosmosDB Storage implementation of PipelineStorage."""

import json
import logging
import math
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from io import BytesIO, StringIO
from typing import Any

import pandas as pd
from azure.cosmos import ContainerProxy, CosmosClient, DatabaseProxy
from azure.cosmos.exceptions import CosmosResourceNotFoundError, CosmosHttpResponseError
from azure.cosmos.partition_key import PartitionKey
from azure.identity import DefaultAzureCredential

from graphrag.logger.progress import Progress
from graphrag.storage.pipeline_storage import (
    PipelineStorage,
    get_timestamp_formatted_with_local_tz,
)

logger = logging.getLogger(__name__)


class CosmosDBPipelineStorage(PipelineStorage):
    """The CosmosDB-Storage Implementation."""

    _cosmos_client: CosmosClient
    _database_client: DatabaseProxy | None
    _container_client: ContainerProxy | None
    _cosmosdb_account_url: str | None
    _connection_string: str | None
    _database_name: str
    _container_name: str
    _encoding: str
    _no_id_prefixes: list[str]

    def __init__(
        self,
        database_name: str | None = None,
        container_name: str | None = None,
        cosmosdb_account_url: str | None = None,
        connection_string: str | None = None,
        encoding: str = "utf-8",
        type: str | None = None,  # Accept type parameter for GraphRAG compatibility
        base_dir: str | None = None,  # GraphRAG passes base_dir instead of database_name
        storage_account_blob_url: str | None = None,  # GraphRAG parameter
        **kwargs  # Accept any additional parameters
    ) -> None:
        """Create a CosmosDB storage instance."""
        logger.info("Creating cosmosdb storage")
        
        # Handle GraphRAG parameters - base_dir is used as database_name
        if base_dir is not None and database_name is None:
            database_name = base_dir
        if database_name is None:
            raise ValueError("Either database_name or base_dir must be provided")
        if container_name is None:
            raise ValueError("container_name must be provided")
        if not database_name:
            msg = "No base_dir provided for database name"
            raise ValueError(msg)
        if connection_string is None and cosmosdb_account_url is None:
            msg = "connection_string or cosmosdb_account_url is required."
            raise ValueError(msg)

        if connection_string:
            self._cosmos_client = CosmosClient.from_connection_string(connection_string)
        else:
            if cosmosdb_account_url is None:
                msg = (
                    "Either connection_string or cosmosdb_account_url must be provided."
                )
                raise ValueError(msg)
            self._cosmos_client = CosmosClient(
                url=cosmosdb_account_url,
                credential=DefaultAzureCredential(),
            )
        self._encoding = kwargs.get("encoding", "utf-8")
        self._database_name = database_name
        self._connection_string = connection_string
        self._cosmosdb_account_url = cosmosdb_account_url
        self._container_name = container_name
        self._cosmosdb_account_name = (
            cosmosdb_account_url.split("//")[1].split(".")[0]
            if cosmosdb_account_url
            else None
        )
        self._no_id_prefixes = []
        logger.debug(
            "creating cosmosdb storage with account: %s and database: %s and container: %s",
            self._cosmosdb_account_name,
            self._database_name,
            self._container_name,
        )
        self._create_database()
        # Container is created lazily on first access (when data is written/read)
        # This prevents creating empty containers for unused storage paths
        self._container_client = None

    def _create_database(self) -> None:
        """Create the database if it doesn't exist."""
        self._database_client = self._cosmos_client.create_database_if_not_exists(
            id=self._database_name
        )

    def _delete_database(self) -> None:
        """Delete the database if it exists."""
        if self._database_client:
            self._database_client = self._cosmos_client.delete_database(
                self._database_client
            )
        self._container_client = None

    def _create_container(self) -> None:
        """Create a container for the current container name if it doesn't exist.
        
        This is called lazily - containers are only created when first accessed
        (on first read/write operation), not during initialization.
        """
        if self._container_client is not None:
            return  # Container already exists
        
        partition_key = PartitionKey(path="/id", kind="Hash")
        if self._database_client:
            self._container_client = (
                self._database_client.create_container_if_not_exists(
                    id=self._container_name,
                    partition_key=partition_key,
                )
            )
            logger.debug("Created container: %s", self._container_name)

    def _ensure_container(self) -> None:
        """Ensure the container exists before performing operations.
        
        This is a convenience method that checks if container exists and creates it if needed.
        """
        if self._container_client is None:
            self._create_container()

    def _delete_container(self) -> None:
        """Delete the container with the current container name if it exists."""
        if self._database_client and self._container_client:
            self._container_client = self._database_client.delete_container(
                self._container_client
            )

    def find(
        self,
        file_pattern: re.Pattern[str],
        base_dir: str | None = None,
        file_filter: dict[str, Any] | None = None,
        max_count=-1,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Find documents in a Cosmos DB container using a file pattern regex and custom file filter (optional).

        Params:
            base_dir: The name of the base directory (not used in Cosmos DB context).
            file_pattern: The file pattern to use.
            file_filter: A dictionary of key-value pairs to filter the documents.
            max_count: The maximum number of documents to return. If -1, all documents are returned.

        Returns
        -------
            An iterator of document IDs and their corresponding regex matches.
        """
        base_dir = base_dir or ""
        logger.info(
            "search container %s for individual documents matching %s",
            self._container_name,
            file_pattern.pattern,
        )
        if not self._database_client:
            return
        self._ensure_container()
        if not self._container_client:
            return

        def item_filter(item: dict[str, Any]) -> bool:
            if file_filter is None:
                return True
            return all(
                re.search(value, item.get(key, ""))
                for key, value in file_filter.items()
            )

        try:
            # Query individual documents based on case_id and is_text
            if file_filter and "case_id" in file_filter:
                case_id = file_filter["case_id"]
                query = "SELECT * FROM c WHERE c.case_id = @case_id"
                parameters = [{"name": "@case_id", "value": case_id}]
                if "is_text" in file_filter:
                    query += " AND c.is_text = @is_text"
                    parameters.append({"name": "@is_text", "value": file_filter["is_text"]})
                items = self._container_client.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True
                )
            else:
                query = "SELECT * FROM c"
                items = self._container_client.query_items(
                    query=query,
                    enable_cross_partition_query=True
                )
            
            num_loaded = 0
            num_filtered = 0
            for item in items:
                filename = item.get("id", "")
                match = file_pattern.search(filename)
                if match and item_filter(item):
                    group = match.groupdict()
                    # Include document metadata in the group
                    group.update({
                        "original_filename": item.get("original_filename", ""),
                        "case_id": item.get("case_id", ""),
                        "document_id": item.get("document_id", ""),
                        "firm_id": item.get("firm_id", ""),
                        "content_type": item.get("content_type", ""),
                        "is_text": item.get("is_text", ""),
                        "is_indexed": item.get("is_indexed", False),
                        "created_at": item.get("created_at", ""),
                    })
                    yield (filename, group)
                    num_loaded += 1
                    if max_count > 0 and num_loaded >= max_count:
                        break
                else:
                    num_filtered += 1

                progress_status = _create_progress_status(
                    num_loaded, num_filtered, num_loaded + num_filtered
                )
                logger.debug(
                    "Progress: %s (%d/%d completed)",
                    progress_status.description,
                    progress_status.completed_items,
                    progress_status.total_items,
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "An error occurred while searching for individual documents in Cosmos DB."
            )

    async def get(
        self, key: str, as_bytes: bool | None = None, encoding: str | None = None
    ) -> Any:
        """Fetch individual documents."""
        try:
            if not self._database_client:
                return None
            self._ensure_container()
            if not self._container_client:
                return None
            if as_bytes:
                prefix = self._get_prefix(key)
                query = f"SELECT * FROM c WHERE STARTSWITH(c.id, '{prefix}')"  # noqa: S608
                queried_items = self._container_client.query_items(
                    query=query, enable_cross_partition_query=True
                )
                items_list = list(queried_items)
                
                # ✅ CRITICAL: Reassemble chunked items BEFORE creating DataFrame
                reassembled_items = {}
                processed_ids = set()
                
                for item in items_list:
                    item_id = item.get("id", "")
                    base_id = item_id.split(":")[1] if ":" in item_id else item_id
                    
                    # Check if this is a chunked item (has _part2 in ID)
                    if "_part2" in base_id:
                        # Extract original ID (remove _part2)
                        original_base_id = base_id.replace("_part2", "")
                        original_id = f"{prefix}:{original_base_id}"
                        
                        if original_id not in reassembled_items:
                            # Initialize - will be populated from part1
                            reassembled_items[original_id] = None
                            processed_ids.add(original_id)
                        
                        # This is part 2 - store for later merging
                        if original_id not in reassembled_items or reassembled_items[original_id] is None:
                            # Create a marker to indicate we have part2
                            reassembled_items[f"{original_id}_part2"] = item
                        else:
                            # We already have part1, merge now
                            part1_item = reassembled_items[original_id]
                            merged_item = self._merge_chunked_item(part1_item, item, base_id)
                            reassembled_items[original_id] = merged_item
                    else:
                        # Regular (non-chunked) item or part 1
                        if item_id not in reassembled_items:
                            # Check if there's a part2 for this item
                            part2_id = f"{item_id}_part2"
                            if part2_id in reassembled_items:
                                # We have part2, merge now
                                part2_item = reassembled_items[part2_id]
                                merged_item = self._merge_chunked_item(item, part2_item, base_id)
                                reassembled_items[item_id] = merged_item
                                del reassembled_items[part2_id]
                            else:
                                # No part2, store as-is
                                reassembled_items[item_id] = item
                                processed_ids.add(item_id)
                
                # Build final items list (one per original ID)
                final_items = []
                for item_id, item_data in reassembled_items.items():
                    if item_id.endswith("_part2"):
                        # Skip part2 items that weren't merged (shouldn't happen, but safety check)
                        continue
                    
                    if item_data is None:
                        # Item was marked but never populated (shouldn't happen)
                        continue
                    
                    # Extract base ID (remove prefix)
                    base_id = item_id.split(":")[1] if ":" in item_id else item_id
                    item_data["id"] = base_id
                    final_items.append(item_data)
                
                # Convert to DataFrame
                items_json_str = json.dumps(final_items)
                items_df = pd.read_json(
                    StringIO(items_json_str), orient="records", lines=False
                )

                # Drop the "id" column if the original dataframe does not include it
                # TODO: Figure out optimal way to handle missing id keys in input dataframes
                if prefix in self._no_id_prefixes:
                    items_df.drop(columns=["id"], axis=1, inplace=True)

                return items_df.to_parquet()
            
            # Handle .txt files for individual documents
            if key.endswith('.txt') and not as_bytes:
                try:
                    doc = self._container_client.read_item(
                        item=key,
                        partition_key=key
                    )
                    content = doc.get("content", "")
                    logger.info(f"Retrieved individual document: {key} ({len(content)} chars)")
                    return content
                except Exception as e:
                    logger.warning(f"Could not retrieve individual document {key}: {e}")
                    return None
            
            # Check if this is a chunked cache entry
            try:
                item = self._container_client.read_item(item=key, partition_key=key)
                
                # Check if this is a chunked document (has _original_id and _part_number)
                original_id = item.get("_original_id")
                part_number = item.get("_part_number")
                total_parts = item.get("_total_parts")
                
                if original_id and part_number:
                    # This is a chunked document - need to reassemble all parts
                    logger.info(f"Reassembling chunked cache item {key} (part {part_number}/{total_parts})")
                    
                    # Query all parts for this original_id
                    query = f"SELECT * FROM c WHERE c._original_id = '{original_id}' ORDER BY c._part_number"  # noqa: S608
                    parts = list(self._container_client.query_items(
                        query=query,
                        enable_cross_partition_query=True
                    ))
                    
                    if not parts:
                        logger.warning(f"No parts found for chunked cache item {original_id}")
                        return None
                    
                    # Reassemble body from parts
                    body_parts = []
                    for part_num in range(1, total_parts + 1):
                        part_item = next((p for p in parts if p.get("_part_number") == part_num), None)
                        if part_item:
                            body_parts.append(part_item.get("body", ""))
                        else:
                            logger.warning(f"Missing part {part_num} for cache item {original_id}")
                    
                    # Join body parts and parse as JSON
                    reassembled_body_str = "".join(body_parts)
                    try:
                        reassembled_body = json.loads(reassembled_body_str)
                        return json.dumps(reassembled_body)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse reassembled cache item {original_id}: {e}")
                        return None
                else:
                    # Regular (non-chunked) document
                    item_body = item.get("body")
                    return json.dumps(item_body)
            except CosmosResourceNotFoundError:
                # Item doesn't exist - might be chunked, try to find parts
                query = f"SELECT * FROM c WHERE c._original_id = '{key}'"  # noqa: S608
                parts = list(self._container_client.query_items(
                    query=query,
                    enable_cross_partition_query=True
                ))
                
                if parts:
                    # Found parts - reassemble
                    original_id = parts[0].get("_original_id")
                    total_parts = parts[0].get("_total_parts", len(parts))
                    
                    body_parts = []
                    for part_num in range(1, total_parts + 1):
                        part_item = next((p for p in parts if p.get("_part_number") == part_num), None)
                        if part_item:
                            body_parts.append(part_item.get("body", ""))
                    
                    if body_parts:
                        reassembled_body_str = "".join(body_parts)
                        try:
                            reassembled_body = json.loads(reassembled_body_str)
                            return json.dumps(reassembled_body)
                        except json.JSONDecodeError:
                            return None
                
                # Item not found and not chunked
                return None
        except Exception:  # noqa: BLE001
            logger.warning("Error reading item %s", key)
            return None

    async def set(self, key: str, value: Any, encoding: str | None = None) -> None:
        """Insert or overwrite the contents of a file into a CosmosDB container for the given filename key.

        Handles parquet datasets (entities, documents, relationships) and JSON (cache/stats).
        Ensures GraphRAG prefix consistency and safe overwriting.
        """

        try:
            if not self._database_client:
                raise ValueError("Database not initialized")
            # Create container lazily on first write operation
            self._ensure_container()
            if not self._container_client:
                raise ValueError("Container could not be created")

            # --- Case 1: Parquet-based datasets (GraphRAG core tables) ---
            if isinstance(value, bytes):
                prefix = self._get_prefix(key)
                value_df = pd.read_parquet(BytesIO(value))
                value_json = value_df.to_json(orient="records", lines=False, force_ascii=False)

                if not value_json:
                    logger.error("Error converting output %s to json", key)
                    return

                cosmosdb_item_list = json.loads(value_json)
                MAX_DOCUMENT_SIZE = 2 * 1024 * 1024  # 2MB
                SAFETY_MARGIN = 50 * 1024  # 50KB safety margin for CosmosDB overhead

                for index, cosmosdb_item in enumerate(cosmosdb_item_list):
                    raw_id = cosmosdb_item.get("id", "").strip() if cosmosdb_item.get("id") else ""

                    # ✅ Always keep GraphRAG dataset prefixes (documents:, entities:, relationships:)
                    if raw_id.startswith(prefix + ":"):
                        prefixed_id = raw_id
                    elif raw_id:
                        prefixed_id = f"{prefix}:{raw_id}"
                    else:
                        prefixed_id = f"{prefix}:{index}"

                    cosmosdb_item["id"] = prefixed_id

                    # Check if item exceeds 2MB (with safety margin)
                    item_json_str = json.dumps(cosmosdb_item, ensure_ascii=False)
                    item_size = len(item_json_str.encode('utf-8'))
                    
                    needs_chunking = item_size > (MAX_DOCUMENT_SIZE - SAFETY_MARGIN)
                    
                    if needs_chunking:
                        # Chunk the item
                        self._chunk_and_store_item(cosmosdb_item, prefixed_id, MAX_DOCUMENT_SIZE)
                    else:
                        # Item fits in one document (or is close to limit)
                        # ✅ UPDATE CLEANUP: Delete old part2 if it exists (in case it was previously chunked)
                        part2_id = f"{prefixed_id}_part2"
                        try:
                            self._container_client.delete_item(item=part2_id, partition_key=part2_id)
                            logger.debug(f"[CLEANUP] Deleted old part2 (item now fits in one document): {part2_id}")
                        except Exception:
                            pass  # Part2 doesn't exist, that's fine
                        
                        # Try to upsert, but fall back to chunking if it fails due to size
                        try:
                            self._container_client.upsert_item(body=cosmosdb_item)
                            logger.info(f"Upserted item: {prefixed_id}")
                        except Exception as e:
                            # Check if it's a size-related error
                            is_size_error = False
                            if isinstance(e, CosmosHttpResponseError):
                                if e.status_code == 413:  # Request Entity Too Large
                                    is_size_error = True
                            elif "Connection aborted" in str(e) or "Remote end closed" in str(e):
                                # Connection abort often indicates size limit exceeded
                                is_size_error = True
                            
                            if is_size_error:
                                logger.warning(f"Item {prefixed_id} exceeded 2MB on upsert (size check missed it, {item_size} bytes), falling back to chunking...")
                                # Fall back to chunking
                                self._chunk_and_store_item(cosmosdb_item, prefixed_id, MAX_DOCUMENT_SIZE)
                            else:
                                # Re-raise if it's not a size error
                                raise

            # --- Case 2: Non-parquet JSON files (cache, stats.json, context.json, etc.) ---
            else:
                MAX_DOCUMENT_SIZE = 2 * 1024 * 1024  # 2MB
                SAFETY_MARGIN = 50 * 1024  # 50KB safety margin for CosmosDB overhead
                
                # Parse the JSON value
                body_data = json.loads(value)
                
                # Create base document to calculate metadata size
                base_item = {
                    "id": key,
                    "body": {},
                }
                base_size = len(json.dumps(base_item, ensure_ascii=False).encode('utf-8'))
                
                # Create full document to check total size
                cosmosdb_item = {
                    "id": key,
                    "body": body_data,
                }
                full_size = len(json.dumps(cosmosdb_item, ensure_ascii=False).encode('utf-8'))
                
                # Check if it exceeds 2MB (with safety margin)
                if full_size > (MAX_DOCUMENT_SIZE - SAFETY_MARGIN):
                    logger.warning(f"Cache item {key} exceeds 2MB ({full_size} bytes), attempting to chunk...")
                    
                    # Calculate available space for body
                    available_body_size = MAX_DOCUMENT_SIZE - base_size - 1024  # 1KB safety margin
                    
                    if available_body_size <= 0:
                        logger.error(f"Cache item {key} metadata alone exceeds 2MB limit")
                        raise ValueError(f"Cache item {key} metadata alone exceeds 2MB limit")
                    
                    # Convert body to JSON string for chunking
                    body_json_str = json.dumps(body_data, ensure_ascii=False)
                    body_bytes = body_json_str.encode('utf-8')
                    body_size = len(body_bytes)
                    
                    # Calculate number of chunks
                    num_chunks = math.ceil(body_size / available_body_size) if body_size > 0 else 1
                    
                    logger.info(f"Splitting cache item {key} into {num_chunks} parts...")
                    
                    # Store chunks
                    for part_num in range(1, num_chunks + 1):
                        start_byte = (part_num - 1) * available_body_size
                        end_byte = min(part_num * available_body_size, body_size)
                        
                        chunk_bytes = body_bytes[start_byte:end_byte]
                        
                        # Decode with UTF-8 error handling
                        try:
                            chunk_body_str = chunk_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            # Find last complete character boundary
                            while len(chunk_bytes) > 0:
                                try:
                                    chunk_body_str = chunk_bytes.decode('utf-8')
                                    break
                                except UnicodeDecodeError:
                                    chunk_bytes = chunk_bytes[:-1]
                            else:
                                chunk_body_str = chunk_bytes.decode('utf-8', errors='replace')
                        
                        # Create part ID
                        part_id = f"{key}_part{part_num}"
                        
                        # Store chunk (body is stored as string chunk, not parsed JSON)
                        part_item = {
                            "id": part_id,
                            "body": chunk_body_str,  # Store as string chunk
                            "_original_id": key,
                            "_part_number": part_num,
                            "_total_parts": num_chunks,
                        }
                        
                        part_size = len(json.dumps(part_item, ensure_ascii=False).encode('utf-8'))
                        if part_size > MAX_DOCUMENT_SIZE:
                            raise ValueError(f"Cache item part {part_id} still exceeds 2MB after chunking")
                        
                        self._container_client.upsert_item(body=part_item)
                        logger.info(f"Upserted cache item part {part_num}/{num_chunks}: {part_id}")
                else:
                    # Item fits in one document (or is close to limit)
                    # Try to upsert, but fall back to chunking if it fails due to size/timeout
                    try:
                        self._container_client.upsert_item(body=cosmosdb_item)
                        logger.info(f"Upserted single item: {key}")
                    except Exception as e:
                        # Check if it's a size-related error (timeout often indicates size issue)
                        is_size_error = False
                        if isinstance(e, CosmosHttpResponseError):
                            if e.status_code == 413:  # Request Entity Too Large
                                is_size_error = True
                        elif "Connection aborted" in str(e) or "Remote end closed" in str(e) or "timed out" in str(e).lower():
                            # Connection abort or timeout often indicates size limit exceeded
                            is_size_error = True
                        
                        if is_size_error:
                            logger.warning(f"Cache item {key} exceeded 2MB on upsert (size check missed it, {full_size} bytes), falling back to chunking...")
                            # Fall back to chunking - re-enter chunking logic
                            available_body_size = MAX_DOCUMENT_SIZE - base_size - 1024  # 1KB safety margin
                            
                            if available_body_size <= 0:
                                logger.error(f"Cache item {key} metadata alone exceeds 2MB limit")
                                raise ValueError(f"Cache item {key} metadata alone exceeds 2MB limit")
                            
                            # Convert body to JSON string for chunking
                            body_json_str = json.dumps(body_data, ensure_ascii=False)
                            body_bytes = body_json_str.encode('utf-8')
                            body_size = len(body_bytes)
                            
                            # Calculate number of chunks
                            num_chunks = math.ceil(body_size / available_body_size) if body_size > 0 else 1
                            
                            logger.info(f"Splitting cache item {key} into {num_chunks} parts...")
                            
                            # Store chunks
                            for part_num in range(1, num_chunks + 1):
                                start_byte = (part_num - 1) * available_body_size
                                end_byte = min(part_num * available_body_size, body_size)
                                
                                chunk_bytes = body_bytes[start_byte:end_byte]
                                
                                # Decode with UTF-8 error handling
                                try:
                                    chunk_body_str = chunk_bytes.decode('utf-8')
                                except UnicodeDecodeError:
                                    while len(chunk_bytes) > 0:
                                        try:
                                            chunk_body_str = chunk_bytes.decode('utf-8')
                                            break
                                        except UnicodeDecodeError:
                                            chunk_bytes = chunk_bytes[:-1]
                                    else:
                                        chunk_body_str = chunk_bytes.decode('utf-8', errors='replace')
                                
                                part_id = f"{key}_part{part_num}"
                                
                                part_item = {
                                    "id": part_id,
                                    "body": chunk_body_str,
                                    "_original_id": key,
                                    "_part_number": part_num,
                                    "_total_parts": num_chunks,
                                }
                                
                                part_size = len(json.dumps(part_item, ensure_ascii=False).encode('utf-8'))
                                if part_size > MAX_DOCUMENT_SIZE:
                                    raise ValueError(f"Cache item part {part_id} still exceeds 2MB after chunking")
                                
                                self._container_client.upsert_item(body=part_item)
                                logger.info(f"Upserted cache item part {part_num}/{num_chunks}: {part_id}")
                        else:
                            # Re-raise if it's not a size error
                            raise

        except Exception:
            logger.exception("Error writing item %s", key)

        

    async def has(self, key: str) -> bool:
        """Check if the contents of the given filename key exist in the cosmosdb storage."""
        if not self._database_client:
            return False
        self._ensure_container()
        if not self._container_client:
            return False
        if ".parquet" in key:
            prefix = self._get_prefix(key)
            query = f"SELECT * FROM c WHERE STARTSWITH(c.id, '{prefix}')"  # noqa: S608
            queried_items = self._container_client.query_items(
                query=query, enable_cross_partition_query=True
            )
            return len(list(queried_items)) > 0
        query = f"SELECT * FROM c WHERE c.id = '{key}'"  # noqa: S608
        queried_items = self._container_client.query_items(
            query=query, enable_cross_partition_query=True
        )
        return len(list(queried_items)) == 1

    async def delete(self, key: str) -> None:
        """Delete all cosmosdb items belonging to the given filename key."""
        if not self._database_client:
            return
        self._ensure_container()
        if not self._container_client:
            return
        try:
            if ".parquet" in key:
                prefix = self._get_prefix(key)
                query = f"SELECT * FROM c WHERE STARTSWITH(c.id, '{prefix}')"  # noqa: S608
                queried_items = self._container_client.query_items(
                    query=query, enable_cross_partition_query=True
                )
                for item in queried_items:
                    self._container_client.delete_item(
                        item=item["id"], partition_key=item["id"]
                    )
            else:
                self._container_client.delete_item(item=key, partition_key=key)
        except CosmosResourceNotFoundError:
            return
        except Exception:
            logger.exception("Error deleting item %s", key)

    async def clear(self) -> None:
        """Clear all contents from storage.

        # This currently deletes the database, including all containers and data within it.
        # TODO: We should decide what granularity of deletion is the ideal behavior (e.g. delete all items within a container, delete the current container, delete the current database)
        """
        self._delete_database()

    def keys(self) -> list[str]:
        """Return the keys in the storage."""
        msg = "CosmosDB storage does yet not support listing keys."
        raise NotImplementedError(msg)

    def child(self, name: str | None) -> PipelineStorage:
        """Create a child storage instance.
        
        For CosmosDB, creates a new storage instance with a modified container name
        that includes the child path. This ensures that delta and previous data are
        stored in separate containers during incremental indexing.
        
        Examples:
            - update_output_5678 -> child('20231103-120000') -> update_output_5678_20231103-120000
            - update_output_5678_20231103-120000 -> child('delta') -> update_output_5678_20231103-120000_delta
            - update_output_5678_20231103-120000 -> child('previous') -> update_output_5678_20231103-120000_previous
        """
        if name is None:
            return self
        
        # Append child name to container name to create logical separation
        # This matches the filesystem behavior where child() creates subdirectories
        child_container_name = f"{self._container_name}_{name}"
        
        return CosmosDBPipelineStorage(
            database_name=self._database_name,
            container_name=child_container_name,
            cosmosdb_account_url=self._cosmosdb_account_url,
            connection_string=self._connection_string,
            encoding=self._encoding,
        )

    def _get_prefix(self, key: str) -> str:
        """Get the prefix of the filename key."""
        return key.split(".")[0]
    
    def _chunk_and_store_item(self, cosmosdb_item: dict, prefixed_id: str, max_document_size: int) -> None:
        """
        Chunk a large item and store it as two parts in CosmosDB.
        
        Args:
            cosmosdb_item: The item to chunk
            prefixed_id: The prefixed ID for the item
            max_document_size: Maximum document size (2MB)
        """
        logger.warning(f"Item {prefixed_id} exceeds 2MB, splitting long field...")
        
        # Find the largest field (likely description, attributes, or text_unit_ids)
        largest_field = None
        largest_field_size = 0
        
        for field_name, field_value in cosmosdb_item.items():
            if field_name == "id":
                continue
            field_json = json.dumps(field_value, ensure_ascii=False)
            field_size = len(field_json.encode('utf-8'))
            if field_size > largest_field_size:
                largest_field_size = field_size
                largest_field = field_name
        
        if not largest_field:
            logger.error(f"Could not find field to chunk for {prefixed_id}")
            raise ValueError(f"Item {prefixed_id} exceeds 2MB but no chunkable field found")
        
        # Get the field value
        field_value = cosmosdb_item[largest_field]
        
        # Convert to string for chunking
        if isinstance(field_value, str):
            field_str = field_value
        else:
            # For dict/list, convert to JSON string
            field_str = json.dumps(field_value, ensure_ascii=False)
        
        field_bytes = field_str.encode('utf-8')
        field_size = len(field_bytes)
        
        # Calculate base item size (without the large field)
        base_item = cosmosdb_item.copy()
        base_item.pop(largest_field, None)
        base_item.pop("id", None)
        base_json = json.dumps(base_item, ensure_ascii=False)
        base_size = len(base_json.encode('utf-8'))
        
        # Calculate available size for field chunks
        # Account for: base fields + id + field name overhead
        id_size = len(prefixed_id.encode('utf-8'))
        field_name_overhead = len(f'"{largest_field}":""'.encode('utf-8'))
        available_field_size = max_document_size - base_size - id_size - field_name_overhead - 1024  # 1KB safety
        
        if available_field_size <= 0:
            logger.error(f"Item {prefixed_id} base fields alone exceed 2MB limit")
            raise ValueError(f"Item {prefixed_id} base fields alone exceed 2MB limit")
        
        # Split field into two chunks
        chunk1_size = available_field_size
        chunk1_bytes = field_bytes[:chunk1_size]
        chunk2_bytes = field_bytes[chunk1_size:]
        
        # Decode chunks with UTF-8 error handling
        try:
            chunk1_str = chunk1_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Find last complete character boundary
            while len(chunk1_bytes) > 0:
                try:
                    chunk1_str = chunk1_bytes.decode('utf-8')
                    break
                except UnicodeDecodeError:
                    chunk1_bytes = chunk1_bytes[:-1]
            else:
                chunk1_str = chunk1_bytes.decode('utf-8', errors='replace')
        
        try:
            chunk2_str = chunk2_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Find last complete character boundary
            while len(chunk2_bytes) > 0:
                try:
                    chunk2_str = chunk2_bytes.decode('utf-8')
                    break
                except UnicodeDecodeError:
                    chunk2_bytes = chunk2_bytes[:-1]
            else:
                chunk2_str = chunk2_bytes.decode('utf-8', errors='replace')
        
        # Parse back to original type if needed
        if isinstance(field_value, str):
            chunk1_value = chunk1_str
            chunk2_value = chunk2_str
        else:
            # For dict/list, try to parse (may fail if chunk is incomplete JSON)
            try:
                chunk1_value = json.loads(chunk1_str)
            except:
                chunk1_value = chunk1_str  # Fallback to string
            try:
                chunk2_value = json.loads(chunk2_str)
            except:
                chunk2_value = chunk2_str  # Fallback to string
        
        # Create part 1: All fields + chunk1 of large field
        part1_item = cosmosdb_item.copy()
        part1_item[largest_field] = chunk1_value
        
        # Verify part1 size
        part1_json = json.dumps(part1_item, ensure_ascii=False)
        part1_size = len(part1_json.encode('utf-8'))
        
        if part1_size > max_document_size:
            logger.error(f"Part1 of {prefixed_id} still exceeds 2MB ({part1_size} bytes)")
            raise ValueError(f"Part1 of {prefixed_id} still exceeds 2MB after chunking")
        
        # Create part 2: All fields + chunk2 of large field
        part2_item = cosmosdb_item.copy()
        part2_item["id"] = f"{prefixed_id}_part2"
        part2_item[largest_field] = chunk2_value
        
        # Verify part2 size
        part2_json = json.dumps(part2_item, ensure_ascii=False)
        part2_size = len(part2_json.encode('utf-8'))
        
        if part2_size > max_document_size:
            logger.warning(f"Part2 of {prefixed_id} exceeds 2MB ({part2_size} bytes), may need further splitting")
            # For now, still try to store (CosmosDB will reject if too large)
        
        # ✅ UPDATE CLEANUP: Delete old part2 if it exists (in case this was previously chunked)
        part2_id = f"{prefixed_id}_part2"
        try:
            self._container_client.delete_item(item=part2_id, partition_key=part2_id)
            logger.debug(f"[CLEANUP] Deleted old part2 before update: {part2_id}")
        except Exception:
            pass  # Part2 doesn't exist, that's fine
        
        # Store both parts
        self._container_client.upsert_item(body=part1_item)
        logger.info(f"Upserted item part 1/2: {prefixed_id} ({part1_size} bytes, chunked field: {largest_field})")
        
        self._container_client.upsert_item(body=part2_item)
        logger.info(f"Upserted item part 2/2: {prefixed_id}_part2 ({part2_size} bytes, chunked field: {largest_field})")
    
    def _merge_chunked_item(self, part1_item: dict, part2_item: dict, base_id: str) -> dict:
        """
        Merge two chunked item parts back into a complete item.
        
        Args:
            part1_item: First part of the chunked item
            part2_item: Second part of the chunked item (has _part2 in ID)
            base_id: Base ID for reference
            
        Returns:
            Merged item with original ID and complete fields
        """
        # Start with part1 as base
        merged_item = part1_item.copy()
        
        # Find which field was chunked (the one that differs between part1 and part2)
        # Strategy: Compare fields - the chunked field will have different values
        chunked_field = None
        
        for field_name in part1_item.keys():
            if field_name == "id":
                continue
            
            part1_value = part1_item.get(field_name)
            part2_value = part2_item.get(field_name)
            
            # If values are different, this is likely the chunked field
            if part1_value != part2_value:
                chunked_field = field_name
                break
        
        # If we couldn't find by comparison, check common large fields
        if not chunked_field:
            # Common fields that get chunked: description, attributes, text_unit_ids, full_content
            large_fields = ["description", "attributes", "text_unit_ids", "full_content", "summary", "findings"]
            for field in large_fields:
                if field in part1_item and field in part2_item:
                    chunked_field = field
                    break
        
        if chunked_field:
            # Merge the chunked field
            part1_value = part1_item.get(chunked_field)
            part2_value = part2_item.get(chunked_field)
            
            if isinstance(part1_value, str) and isinstance(part2_value, str):
                # Both are strings - concatenate
                merged_item[chunked_field] = part1_value + part2_value
            elif isinstance(part1_value, list) and isinstance(part2_value, list):
                # Both are lists - concatenate
                merged_item[chunked_field] = part1_value + part2_value
            elif isinstance(part1_value, dict) and isinstance(part2_value, dict):
                # Both are dicts - merge
                merged_item[chunked_field] = {**part1_value, **part2_value}
            else:
                # Different types or one is None - use part1 (shouldn't happen)
                logger.warning(f"Could not merge field {chunked_field} for {base_id}, types don't match")
                merged_item[chunked_field] = part1_value
        else:
            logger.warning(f"Could not identify chunked field for {base_id}, using part1 as-is")
        
        # Ensure ID is the original (remove _part2 if present)
        original_id = merged_item.get("id", "")
        if "_part2" in original_id:
            merged_item["id"] = original_id.replace("_part2", "")
        
        return merged_item

    async def get_creation_date(self, key: str) -> str:
        """Get a value from the cache."""
        try:
            if not self._database_client:
                return ""
            self._ensure_container()
            if not self._container_client:
                return ""
            item = self._container_client.read_item(item=key, partition_key=key)
            return get_timestamp_formatted_with_local_tz(
                datetime.fromtimestamp(item["_ts"], tz=timezone.utc)
            )

        except Exception:  # noqa: BLE001
            logger.warning("Error getting key %s", key)
            return ""


def _create_progress_status(
    num_loaded: int, num_filtered: int, num_total: int
) -> Progress:
    return Progress(
        total_items=num_total,
        completed_items=num_loaded + num_filtered,
        description=f"{num_loaded} files loaded ({num_filtered} filtered)",
    )
