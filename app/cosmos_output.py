"""
CosmosDB Output Container Manager
Handles storage of graph data (entities, relationships, communities) AND vectors
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from app.cosmos_client import get_cosmos_manager
import pandas as pd
import json


class OutputManager:
    """Manages output data (graph + vectors) in CosmosDB."""
    
    def __init__(self):
        self.manager = None  # Will be set by get_firm_managers()
        self.container = None  # Will be set by get_firm_managers()
    
    def _normalize_caseid(self, case_id: str) -> str:
        """Normalize case_id for partition key (numeric cases get .txt extension)."""
        return f"{case_id}.txt" if case_id.isdigit() else case_id

    # ==================== Graph Data Methods ====================

    def store_graph_data(
        self,
        case_id: str,
        data_type: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store graph data (entities, relationships, communities, etc.).
        In case-specific containers, entities with the same title are overwritten
        instead of creating duplicates.
        """
        # Convert to native Python objects
        if isinstance(data, pd.DataFrame):
            data_dict = json.loads(data.to_json(orient='records'))
        elif isinstance(data, list):
            data_dict = data
        else:
            data_dict = data

        # === Special Handling for Entities ===
        if data_type == "entities" and isinstance(data_dict, list):
            for entity in data_dict:
                title_value = entity.get("title", "").strip().lower()
                if not title_value:
                    continue

                # Look for existing entity with same title (case-specific container)
                query = """
                SELECT c.id FROM c 
                WHERE LOWER(c.title) = @title AND c.type = 'entities'
                """
                parameters = [{"name": "@title", "value": title_value}]
                existing = list(
                    self.container.query_items(
                        query=query,
                        parameters=parameters,
                        enable_cross_partition_query=False
                    )
                )

                # Reuse existing ID if found, otherwise create new one
                if existing:
                    existing_id = existing[0]["id"]
                    entity_doc = {
                        "id": existing_id,
                        "title": entity.get("title"),
                        "type": "entities",
                        "data": entity,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                    print(f"[UPDATE] Overwriting entity '{title_value}' (id={existing_id})")
                else:
                    new_id = f"entities-{title_value[:40].replace(' ', '_')}"
                    entity_doc = {
                        "id": new_id,
                        "title": entity.get("title"),
                        "type": "entities",
                        "data": entity,
                        "created_at": datetime.utcnow().isoformat(),
                    }
                    print(f"[INSERT] Creating new entity '{title_value}' (id={new_id})")

                self.container.upsert_item(entity_doc)
            return {"status": "success", "type": data_type, "items": len(data_dict)}

        # === Default Handling for Non-Entity Graph Data ===
        caseid = self._normalize_caseid(case_id)
        doc = {
            "id": f"output-{case_id}-{data_type}",
            "caseid": caseid,  # legacy compatibility
            "case_id": case_id,
            "type": data_type,
            "data": data_dict,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        return self.container.upsert_item(doc)

    def get_graph_data(self, case_id: str, data_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve graph data by type."""
        try:
            doc_id = f"output/{case_id}/{data_type}"
            return self.container.read_item(
                item=doc_id, partition_key=self._normalize_caseid(case_id)
            )
        except CosmosResourceNotFoundError:
            return None

    def get_graph_data_as_dataframe(self, case_id: str, data_type: str) -> Optional[pd.DataFrame]:
        """Retrieve graph data as a pandas DataFrame."""
        doc = self.get_graph_data(case_id, data_type)
        if doc and "data" in doc:
            return pd.DataFrame(doc["data"])
        return None

    def list_graph_data_types(self, case_id: str) -> List[str]:
        """List all available graph data types for a case."""
        query = """
        SELECT c.type FROM c 
        WHERE STARTSWITH(c.id, 'output/')
        AND c.type != 'vector'
        """
        items = self.container.query_items(
            query=query,
            enable_cross_partition_query=False
        )
        return [item["type"] for item in items]

    # ==================== Vector Storage Methods ====================

    def store_vector(
        self,
        case_id: str,
        vector_id: str,
        embedding: List[float],
        entity_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Store a vector embedding."""
        doc = {
            "id": f"vectors-{vector_id}",
            "type": "vector",
            "vector_id": vector_id,
            "embedding": embedding,
            "entity_name": entity_name,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        return self.container.upsert_item(doc)

    def get_vector(self, case_id: str, vector_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a vector by ID."""
        try:
            doc_id = f"vectors/{vector_id}"
            return self.container.read_item(
                item=doc_id, partition_key=doc_id
            )
        except CosmosResourceNotFoundError:
            return None

    def list_vectors(self, case_id: str) -> List[Dict[str, Any]]:
        """List all vectors for a case."""
        query = "SELECT * FROM c WHERE c.type = 'vector'"
        items = self.container.query_items(query=query, enable_cross_partition_query=False)
        return list(items)

    def bulk_store_vectors(self, case_id: str, vectors: List[Dict[str, Any]]) -> int:
        """Store multiple vectors at once."""
        count = 0
        for v in vectors:
            self.store_vector(
                case_id=case_id,
                vector_id=v["vector_id"],
                embedding=v["embedding"],
                entity_name=v.get("entity_name"),
                metadata=v.get("metadata")
            )
            count += 1
        return count

    # ==================== Stats and Metadata ====================

    def store_stats(self, case_id: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Store indexing statistics."""
        return self.store_graph_data(case_id, "stats", stats)

    def get_stats(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve indexing statistics."""
        return self.get_graph_data(case_id, "stats")

    def store_metadata(self, case_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store index metadata."""
        return self.store_graph_data(case_id, "metadata", metadata)

    def get_metadata(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve index metadata."""
        return self.get_graph_data(case_id, "metadata")

    def delete_case_data(self, case_id: str):
        """Delete all data for a case (container-specific)."""
        query = "SELECT c.id FROM c"
        items = self.container.query_items(query=query, enable_cross_partition_query=False)
        for it in items:
            try:
                self.container.delete_item(item=it["id"], partition_key=it["id"])
            except CosmosResourceNotFoundError:
                pass
