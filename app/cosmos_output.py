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
    
    # No longer needed - using /id as partition key in case-specific containers

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
            # Store entities as a single document (like other graph data types)
            # This ensures final entities overwrite raw entities properly
            doc = {
                "id": f"output-{case_id}-{data_type}",
                "case_id": case_id,  # Keep for metadata/queries
                "type": data_type,
                "data": data_dict,
                "created_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            print(f"[INFO] Storing {len(data_dict)} entities for case {case_id}")
            return self.container.upsert_item(doc)

        # === Default Handling for Non-Entity Graph Data ===
        doc = {
            "id": f"output-{case_id}-{data_type}",
            "case_id": case_id,  # Keep for metadata/queries
            "type": data_type,
            "data": data_dict,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        return self.container.upsert_item(doc)

    def get_graph_data(self, case_id: str, data_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve graph data by type."""
        try:
            doc_id = f"output-{case_id}-{data_type}"
            return self.container.read_item(item=doc_id, partition_key=doc_id)
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
            doc_id = f"vectors-{vector_id}"
            return self.container.read_item(item=doc_id, partition_key=doc_id)
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

    def deduplicate_entities_by_title(self, case_id: str) -> Dict[str, Any]:
        """
        Remove duplicate entities by comparing titles and keeping only the final versions.
        This method compares raw_entities with final entities and removes duplicates.
        """
        try:
            # Get all entities from the container (both raw and final)
            query = "SELECT * FROM c WHERE STARTSWITH(c.id, 'entities:')"
            all_entities = list(
                self.container.query_items(
                    query=query,
                    enable_cross_partition_query=True
                )
            )
            
            if not all_entities:
                return {"status": "no_entities", "raw_count": 0, "final_count": 0, "duplicates_removed": 0}
            
            # Separate raw and final entities based on ID pattern
            raw_entities = []
            final_entities = []
            
            for entity in all_entities:
                entity_id = entity.get("id", "")
                # Raw entities have simple numeric IDs like "entities:0", "entities:1"
                # Final entities have UUID IDs like "entities:32dfdc00-8396-484c-8fc8-773c71986eec"
                if ":" in entity_id:
                    id_suffix = entity_id.split(":", 1)[1]
                    # Check if ID suffix is numeric (raw) or UUID (final)
                    if id_suffix.isdigit():
                        raw_entities.append(entity)
                    else:
                        final_entities.append(entity)
                else:
                    # Fallback: treat as raw if no colon in ID
                    raw_entities.append(entity)
            
            if not raw_entities and not final_entities:
                return {"status": "no_entities", "raw_count": 0, "final_count": 0, "duplicates_removed": 0}
            
            # Create title sets for comparison
            raw_titles = set()
            final_titles = set()
            
            for entity in raw_entities:
                title = entity.get("title", "").strip().lower()
                if title:
                    raw_titles.add(title)
            
            for entity in final_entities:
                title = entity.get("title", "").strip().lower()
                if title:
                    final_titles.add(title)
            
            # Find duplicates (entities that exist in both raw and final)
            duplicates = raw_titles.intersection(final_titles)
            
            if duplicates:
                # Remove raw entities that have duplicates in final entities
                removed_count = 0
                
                for entity in raw_entities:
                    title = entity.get("title", "").strip().lower()
                    if title in duplicates:
                        try:
                            # Delete the raw entity document
                            self.container.delete_item(item=entity["id"], partition_key=entity["id"])
                            removed_count += 1
                        except Exception:
                            pass  # Ignore deletion errors
                
                return {
                    "status": "success",
                    "raw_count": len(raw_entities),
                    "final_count": len(final_entities),
                    "duplicates_removed": removed_count,
                    "remaining_raw": len(raw_entities) - removed_count
                }
            else:
                return {
                    "status": "no_duplicates",
                    "raw_count": len(raw_entities),
                    "final_count": len(final_entities),
                    "duplicates_removed": 0
                }
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def deduplicate_relationships(self, case_id: str) -> Dict[str, Any]:
        """
        Remove duplicate relationships by comparing source and target entities.
        This method compares raw_relationships with final relationships and removes duplicates.
        """
        try:
            # Get all relationships from the container (both raw and final)
            query = "SELECT * FROM c WHERE STARTSWITH(c.id, 'relationships:')"
            all_relationships = list(
                self.container.query_items(
                    query=query,
                    enable_cross_partition_query=True
                )
            )
            
            if not all_relationships:
                return {"status": "no_relationships", "raw_count": 0, "final_count": 0, "duplicates_removed": 0}
            
            # Separate raw and final relationships based on ID pattern
            raw_relationships = []
            final_relationships = []
            
            for rel in all_relationships:
                rel_id = rel.get("id", "")
                # Raw relationships have simple numeric IDs like "relationships:0", "relationships:1"
                # Final relationships have UUID IDs like "relationships:32dfdc00-8396-484c-8fc8-773c71986eec"
                if ":" in rel_id:
                    id_suffix = rel_id.split(":", 1)[1]
                    # Check if ID suffix is numeric (raw) or UUID (final)
                    if id_suffix.isdigit():
                        raw_relationships.append(rel)
                    else:
                        final_relationships.append(rel)
                else:
                    # Fallback: treat as raw if no colon in ID
                    raw_relationships.append(rel)
            
            if not raw_relationships and not final_relationships:
                return {"status": "no_relationships", "raw_count": 0, "final_count": 0, "duplicates_removed": 0}
            
            # Create relationship keys for comparison (source + target only)
            raw_keys = set()
            final_keys = set()
            
            for rel in raw_relationships:
                source = rel.get("source", "").strip().lower()
                target = rel.get("target", "").strip().lower()
                key = f"{source}|{target}"
                if source and target:
                    raw_keys.add(key)
            
            for rel in final_relationships:
                source = rel.get("source", "").strip().lower()
                target = rel.get("target", "").strip().lower()
                key = f"{source}|{target}"
                if source and target:
                    final_keys.add(key)
            
            # Find duplicates (relationships that exist in both raw and final)
            duplicates = raw_keys.intersection(final_keys)
            
            if duplicates:
                # Remove raw relationships that have duplicates in final relationships
                removed_count = 0
                
                for rel in raw_relationships:
                    source = rel.get("source", "").strip().lower()
                    target = rel.get("target", "").strip().lower()
                    key = f"{source}|{target}"
                    
                    if key in duplicates:
                        try:
                            # Delete the raw relationship document
                            self.container.delete_item(item=rel["id"], partition_key=rel["id"])
                            removed_count += 1
                        except Exception:
                            pass  # Ignore deletion errors
                
                return {
                    "status": "success",
                    "raw_count": len(raw_relationships),
                    "final_count": len(final_relationships),
                    "duplicates_removed": removed_count,
                    "remaining_raw": len(raw_relationships) - removed_count
                }
            else:
                return {
                    "status": "no_duplicates",
                    "raw_count": len(raw_relationships),
                    "final_count": len(final_relationships),
                    "duplicates_removed": 0
                }
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def deduplicate_raw_data(self, case_id: str) -> Dict[str, Any]:
        """
        Remove duplicate raw entities and relationships by comparing with final versions.
        This is the main method to call after GraphRAG indexing is complete.
        """
        # Deduplicate entities
        entity_result = self.deduplicate_entities_by_title(case_id)
        
        # Deduplicate relationships
        relationship_result = self.deduplicate_relationships(case_id)
        
        # Summary
        total_removed = entity_result.get("duplicates_removed", 0) + relationship_result.get("duplicates_removed", 0)
        
        return {
            "status": "success",
            "case_id": case_id,
            "entities": entity_result,
            "relationships": relationship_result,
            "total_duplicates_removed": total_removed
        }
    
    def delete_case_data(self, case_id: str):
        """Delete all data for a case (container-specific)."""
        query = "SELECT c.id FROM c"
        items = self.container.query_items(query=query, enable_cross_partition_query=False)
        for it in items:
            try:
                self.container.delete_item(item=it["id"], partition_key=it["id"])
            except CosmosResourceNotFoundError:
                pass
