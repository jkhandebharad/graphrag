# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Azure CosmosDB Storage implementation of PipelineStorage."""

import json
import logging
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from io import BytesIO, StringIO
from typing import Any

import pandas as pd
from azure.cosmos import ContainerProxy, CosmosClient, DatabaseProxy
from azure.cosmos.exceptions import CosmosResourceNotFoundError
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
                for item in items_list:
                    item["id"] = item["id"].split(":")[1]

                items_json_str = json.dumps(items_list)

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
            
            item = self._container_client.read_item(item=key, partition_key=key)
            item_body = item.get("body")
            return json.dumps(item_body)
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

                    # ✅ Overwrite if already exists (upsert)
                    self._container_client.upsert_item(body=cosmosdb_item)
                    logger.info(f"Upserted item: {prefixed_id}")

            # --- Case 2: Non-parquet JSON files (cache, stats.json, context.json, etc.) ---
            else:
                cosmosdb_item = {
                    "id": key,
                    "body": json.loads(value),
                }
                self._container_client.upsert_item(body=cosmosdb_item)
                logger.info(f"Upserted single item: {key}")

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
