"""
CosmosDB Output Container Manager
Handles storage of graph data (entities, relationships, communities) AND vectors
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from app.cosmos_client import get_cosmos_manager
from graphrag.utils.storage import load_table_from_storage
from graphrag.storage.cosmosdb_pipeline_storage import CosmosDBPipelineStorage
import pandas as pd
import json
import os
import re


class OutputManager:
    """Manages output data (graph + vectors) in CosmosDB."""
    
    def __init__(self):
        self.manager = None  # Will be set by get_firm_managers()
        self.container = None  # Will be set by get_firm_managers()
    
    # No longer needed - using /id as partition key in case-specific containers

    # ==================== Graph Data Methods ====================

    # Note: GraphRAG handles all writes directly through CosmosDBPipelineStorage
    # We don't need store_graph_data() - GraphRAG writes entities/relationships individually
    
    def get_graph_data(self, case_id: str, data_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve graph data by type."""
        try:
            doc_id = f"output-{case_id}-{data_type}"
            return self.container.read_item(item=doc_id, partition_key=doc_id)
        except CosmosResourceNotFoundError:
            return None

    def _get_graphrag_storage(self, case_id: str, firm_id: str) -> CosmosDBPipelineStorage:
        """
        Create a GraphRAG CosmosDBPipelineStorage instance for querying.
        Uses GraphRAG's native storage interface to query data.
        
        Args:
            case_id: Case identifier for case-specific container
            firm_id: Firm identifier to construct database name (graphrag_{firm_id})
        """
        connection_string = os.getenv("COSMOS_CONNECTION_STRING")
        
        if not connection_string:
            # Build connection string from endpoint and key if not available
            endpoint = os.getenv("COSMOS_ENDPOINT")
            key = os.getenv("COSMOS_KEY")
            if endpoint and key:
                # Format: AccountEndpoint=...;AccountKey=...
                connection_string = f"AccountEndpoint={endpoint};AccountKey={key};"
            else:
                raise ValueError("Either COSMOS_CONNECTION_STRING or COSMOS_ENDPOINT+COSMOS_KEY must be set")
        
        # Database name follows pattern: graphrag_{firm_id}
        database_name = f"graphrag_{firm_id}"
        
        # Container name follows the pattern: output_{case_id}
        container_name = f"output_{case_id}"
        
        return CosmosDBPipelineStorage(
            database_name=database_name,
            container_name=container_name,
            connection_string=connection_string
        )
    
    async def get_graph_data_as_dataframe(self, case_id: str, data_type: str, firm_id: str) -> Optional[pd.DataFrame]:
        """
        Retrieve graph data as a pandas DataFrame using GraphRAG's native storage utility.
        
        Uses GraphRAG's load_table_from_storage() which:
        - Queries individual documents (entities:0, entities:uuid, etc.)
        - Combines them into a DataFrame
        - Handles all GraphRAG storage patterns correctly
        
        Args:
            case_id: Case identifier for case-specific container
            data_type: Type of graph data (entities, relationships, etc.)
            firm_id: Firm identifier to construct database name
        """
        try:
            storage = self._get_graphrag_storage(case_id, firm_id)
            # GraphRAG's utility handles the querying and DataFrame conversion
            return await load_table_from_storage(name=data_type, storage=storage)
        except Exception as e:
            print(f"[ERROR] Failed to load {data_type} from CosmosDB using GraphRAG storage: {e}")
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
    # Note: GraphRAG stores stats/metadata through its own storage interface
    # These get methods remain for potential future use
    
    def get_stats(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve indexing statistics."""
        return self.get_graph_data(case_id, "stats")

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
    
    def deduplicate_vector_embeddings(self, case_id: str) -> Dict[str, Any]:
        """
        Remove raw vector embeddings that have final versions with same text content.
        """
        try:
            # Get vector store containers for this case
            vector_containers = [
                f"output_{case_id}-entity-description",
                f"output_{case_id}-community-full_content", 
                f"output_{case_id}-text_unit-text"
            ]
            
            total_duplicates_removed = 0
            
            for container_name in vector_containers:
                try:
                    container = self.manager.database.get_container_client(container_name)
                    
                    # Query all embeddings
                    query = "SELECT * FROM c"
                    embeddings = list(container.query_items(query, enable_cross_partition_query=True))
                    
                    # Separate raw (numeric ID) from final (UUID ID)
                    raw_embeddings = [e for e in embeddings if e["id"].isdigit()]
                    final_embeddings = [e for e in embeddings if not e["id"].isdigit()]
                    
                    # Find duplicates by text content
                    final_texts = {e["text"]: e for e in final_embeddings}
                    duplicates_to_remove = []
                    
                    for raw_emb in raw_embeddings:
                        if raw_emb["text"] in final_texts:
                            duplicates_to_remove.append(raw_emb["id"])
                    
                    # Delete duplicate raw embeddings
                    for doc_id in duplicates_to_remove:
                        container.delete_item(item=doc_id, partition_key=doc_id)
                    
                    total_duplicates_removed += len(duplicates_to_remove)
                    
                except Exception as container_error:
                    # Container might not exist yet, skip it
                    continue
                    
            return {"status": "success", "duplicates_removed": total_duplicates_removed}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def deduplicate_raw_data(self, case_id: str) -> Dict[str, Any]:
        """
        Remove duplicate raw entities, relationships, and vector embeddings by comparing with final versions.
        This is the main method to call after GraphRAG indexing is complete.
        """
        # Deduplicate entities
        entity_result = self.deduplicate_entities_by_title(case_id)
        
        # Deduplicate relationships
        relationship_result = self.deduplicate_relationships(case_id)
        
        # Deduplicate vector embeddings
        vector_result = self.deduplicate_vector_embeddings(case_id)
        
        # Summary
        total_removed = (
            entity_result.get("duplicates_removed", 0) + 
            relationship_result.get("duplicates_removed", 0) +
            vector_result.get("duplicates_removed", 0)
        )
        
        return {
            "status": "success",
            "case_id": case_id,
            "entities": entity_result,
            "relationships": relationship_result,
            "vector_embeddings": vector_result,
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
    
    def cleanup_raw_data_after_incremental_update(self, case_id: str, firm_id: str) -> Dict[str, Any]:
        """
        Clean up raw (numeric ID) entities, relationships, and embeddings after incremental indexing.
        GraphRAG's update workflows automatically copy final data from delta to output storage,
        so we only need to clean up raw data artifacts.
        
        Process:
        1. Clean raw entities and relationships from delta container
        2. Clean raw embeddings from vector store containers
        
        Args:
            case_id: Case identifier
            firm_id: Firm identifier for database name
            
        Returns:
            Dict with cleanup operation results
        """
        try:
            # Find the latest delta container: update_output_{case_id}_{timestamp}_delta
            # Since containers are created with timestamp, we need to find the latest one
            database = self.manager.database
            
            # List all containers and find ones matching update_output_{case_id}_*_delta pattern
            pattern = re.compile(rf"^update_output_{case_id}_(\d{{8}}-\d{{6}})_delta$")
            
            containers = list(database.list_containers())
            matching_containers = []
            
            for container in containers:
                container_id = container['id']
                match = pattern.match(container_id)
                if match:
                    timestamp = match.group(1)
                    matching_containers.append((timestamp, container_id))
            
            if not matching_containers:
                return {
                    "status": "warning",
                    "case_id": case_id,
                    "raw_entities_removed": 0,
                    "raw_relationships_removed": 0,
                    "raw_embeddings_removed": 0,
                    "message": "No delta container found - raw data cleanup skipped"
                }
            
            # Get the latest container (sort by timestamp descending)
            matching_containers.sort(reverse=True, key=lambda x: x[0])
            delta_container_name = matching_containers[0][1]
            
            delta_container = database.get_container_client(delta_container_name)
            
            # Clean raw entities and relationships from delta container
            raw_entities_removed = self._clean_raw_data_from_container(delta_container, "entities:")
            raw_relationships_removed = self._clean_raw_data_from_container(delta_container, "relationships:")
            
            # Clean raw embeddings from vector store containers
            # Vector store containers store embeddings with entity/relationship IDs
            # Need to remove embeddings that correspond to raw (numeric) entity IDs
            raw_embeddings_removed = self._clean_raw_embeddings_from_vector_stores(case_id, firm_id)
            
            return {
                "status": "success",
                "case_id": case_id,
                "raw_entities_removed": raw_entities_removed,
                "raw_relationships_removed": raw_relationships_removed,
                "raw_embeddings_removed": raw_embeddings_removed,
                "message": f"Raw data cleaned from {delta_container_name} and vector stores. GraphRAG has already copied final data to output container."
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "case_id": case_id,
                "error": str(e)
            }
    
    def _clean_raw_data_from_container(self, container, prefix: str) -> int:
        """
        Clean raw entities or relationships (numeric IDs) from a container.
        
        Args:
            container: CosmosDB container to clean
            prefix: Prefix to filter by (e.g., "entities:", "relationships:")
            
        Returns:
            Number of items removed
        """
        removed_count = 0
        
        try:
            # Query items with the prefix
            query = f"SELECT * FROM c WHERE STARTSWITH(c.id, '{prefix}')"
            items = list(container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            
            for item in items:
                item_id = item.get("id", "")
                if ":" in item_id:
                    id_suffix = item_id.split(":", 1)[1]
                    # Check if ID suffix is numeric (raw) - remove only raw data
                    if id_suffix.isdigit():
                        try:
                            container.delete_item(item=item_id, partition_key=item_id)
                            removed_count += 1
                        except Exception as e:
                            print(f"[WARNING] Failed to delete {item_id}: {e}")
                            continue
            
        except Exception as e:
            print(f"[WARNING] Error cleaning raw data with prefix {prefix}: {e}")
        
        return removed_count
    
    def _clean_raw_embeddings_from_vector_stores(self, case_id: str, firm_id: str) -> int:
        """
        Clean raw embeddings from vector store containers.
        Raw embeddings have IDs that match raw entity IDs (numeric or entities:0, entities:1, etc.)
        
        During incremental indexing, new embeddings are created for merged entities.
        Old embeddings with raw entity IDs need to be removed.
        
        Args:
            case_id: Case identifier
            firm_id: Firm identifier for database name
            
        Returns:
            Total number of raw embeddings removed across all vector containers
        """
        total_removed = 0
        vector_containers = [
            f"output_{case_id}-entity-description",
            f"output_{case_id}-community-full_content",
            f"output_{case_id}-text_unit-text"
        ]
        
        if not self.manager or not self.manager.database:
            print(f"[WARNING] Cannot clean embeddings - manager or database not initialized")
            return 0
        
        try:
            for container_name in vector_containers:
                try:
                    container = self.manager.database.get_container_client(container_name)
                    
                    # Query all embeddings in this container
                    # The id field is used as both document ID and partition key
                    query = "SELECT * FROM c"
                    embeddings = list(container.query_items(
                        query=query,
                        enable_cross_partition_query=True
                    ))
                    
                    removed_from_container = 0
                    
                    for emb in embeddings:
                        # Get the document ID (which matches the entity/relationship/text_unit ID)
                        emb_id = emb.get("id", "")
                        
                        if not emb_id:
                            continue
                        
                        # Check if this embedding ID corresponds to a raw entity/relationship/text_unit
                        # Raw IDs can be:
                        # - Numeric only: "0", "1", "2"
                        # - With prefix: "entities:0", "relationships:1", "text_units:2"
                        # - Final IDs are UUIDs: "entities:51d4a181-c2e7-46f9-9896-81c784dbacec"
                        
                        is_raw = False
                        
                        if ":" in emb_id:
                            # Has prefix like "entities:0" or "entities:uuid"
                            id_part = emb_id.split(":", 1)[1]
                            # If purely numeric, it's raw; if has hyphens and is long (>20 chars), it's UUID (final)
                            if id_part.isdigit():
                                is_raw = True
                            # UUIDs have hyphens and are 36 chars: "51d4a181-c2e7-46f9-9896-81c784dbacec"
                            elif "-" in id_part and len(id_part) > 20:
                                is_raw = False  # It's a UUID, keep it
                        else:
                            # No prefix - check if it's purely numeric (raw)
                            if emb_id.isdigit():
                                is_raw = True
                        
                        if is_raw:
                            try:
                                # Delete using the document ID as partition key (CosmosDB uses id field as partition key)
                                container.delete_item(item=emb_id, partition_key=emb_id)
                                removed_from_container += 1
                            except Exception as e:
                                print(f"[WARNING] Failed to delete embedding {emb_id} from {container_name}: {e}")
                                continue
                    
                    if removed_from_container > 0:
                        print(f"[MERGE] Cleaned {removed_from_container} raw embeddings from {container_name}")
                    total_removed += removed_from_container
                    
                except Exception as container_error:
                    # Container might not exist or error accessing it
                    print(f"[WARNING] Could not clean embeddings from {container_name}: {container_error}")
                    continue
                    
        except Exception as e:
            print(f"[WARNING] Error cleaning raw embeddings from vector stores: {e}")
            import traceback
            traceback.print_exc()
        
        return total_removed
    
    def _get_graphrag_storage_for_update(self, case_id: str, firm_id: str, container_name: str) -> CosmosDBPipelineStorage:
        """
        Create a GraphRAG CosmosDBPipelineStorage instance for update_output container.
        
        Args:
            case_id: Case identifier
            firm_id: Firm identifier to construct database name
            container_name: Container name (e.g., update_output_{case_id})
        """
        connection_string = os.getenv("COSMOS_CONNECTION_STRING")
        
        if not connection_string:
            endpoint = os.getenv("COSMOS_ENDPOINT")
            key = os.getenv("COSMOS_KEY")
            if endpoint and key:
                connection_string = f"AccountEndpoint={endpoint};AccountKey={key};"
            else:
                raise ValueError("Either COSMOS_CONNECTION_STRING or COSMOS_ENDPOINT+COSMOS_KEY must be set")
        
        database_name = f"graphrag_{firm_id}"
        
        return CosmosDBPipelineStorage(
            database_name=database_name,
            container_name=container_name,
            connection_string=connection_string
        )