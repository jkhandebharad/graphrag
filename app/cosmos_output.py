"""
CosmosDB Output Container Manager
Handles storage of graph data (entities, relationships, communities) AND vectors
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from app.cosmos_client import get_cosmos_manager
import pandas as pd


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
        
        Args:
            case_id: The case identifier
            data_type: Type of data (entities, relationships, communities, etc.)
            data: The actual data (will be converted to JSON-compatible format)
            metadata: Additional metadata
            
        Returns:
            The stored document
        """
        import json
        
        # Convert pandas DataFrame to dict if needed
        # Use to_json() then parse to ensure numpy types are converted to native Python types
        if isinstance(data, pd.DataFrame):
            data_dict = json.loads(data.to_json(orient='records'))
        elif isinstance(data, list):
            data_dict = data
        else:
            data_dict = data
        
        # CosmosDB IDs cannot contain / : # ? or \ characters
        caseid = self._normalize_caseid(case_id)
        doc = {
            "id": f"output-{case_id}-{data_type}",
            "caseid": caseid,  # Partition key
            "case_id": case_id,  # Original for queries/metadata
            "type": data_type,
            "data": data_dict,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        return self.container.upsert_item(doc)
    
    def get_graph_data(self, case_id: str, data_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve graph data by type.
        
        Args:
            case_id: The case identifier
            data_type: Type of data to retrieve
            
        Returns:
            The document containing the data
        """
        try:
            doc_id = f"output/{case_id}/{data_type}"
            return self.container.read_item(item=doc_id, partition_key=self._normalize_caseid(case_id))
        except CosmosResourceNotFoundError:
            return None
    
    def get_graph_data_as_dataframe(self, case_id: str, data_type: str) -> Optional[pd.DataFrame]:
        """
        Retrieve graph data as a pandas DataFrame.
        
        Args:
            case_id: The case identifier
            data_type: Type of data to retrieve
            
        Returns:
            DataFrame containing the data
        """
        doc = self.get_graph_data(case_id, data_type)
        if doc and "data" in doc:
            return pd.DataFrame(doc["data"])
        return None
    
    def list_graph_data_types(self, case_id: str) -> List[str]:
        """List all available graph data types for a case."""
        query = """
        SELECT c.type FROM c 
        WHERE c.case_id = @case_id 
        AND STARTSWITH(c.id, 'output/')
        AND c.type != 'vector'
        """
        
        items = self.container.query_items(
            query=query,
            parameters=[{"name": "@case_id", "value": case_id}],
            partition_key=self._normalize_caseid(case_id),
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
        """
        Store a vector embedding.
        
        Args:
            case_id: The case identifier
            vector_id: Unique identifier for the vector
            embedding: The vector embedding (list of floats)
            entity_name: Name of the entity this vector represents
            metadata: Additional metadata
            
        Returns:
            The stored vector document
        """
        # CosmosDB IDs cannot contain / : # ? or \ characters
        doc = {
            "id": f"vectors-{case_id}-{vector_id}",
            "case_id": case_id,
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
            doc_id = f"vectors/{case_id}/{vector_id}"
            return self.container.read_item(item=doc_id, partition_key=self._normalize_caseid(case_id))
        except CosmosResourceNotFoundError:
            return None
    
    def list_vectors(self, case_id: str) -> List[Dict[str, Any]]:
        """List all vectors for a case."""
        query = """
        SELECT * FROM c 
        WHERE c.case_id = @case_id 
        AND c.type = 'vector'
        """
        
        items = self.container.query_items(
            query=query,
            parameters=[{"name": "@case_id", "value": case_id}],
            partition_key=self._normalize_caseid(case_id),
            enable_cross_partition_query=False
        )
        
        return list(items)
    
    def bulk_store_vectors(
        self,
        case_id: str,
        vectors: List[Dict[str, Any]]
    ) -> int:
        """
        Store multiple vectors at once.
        
        Args:
            case_id: The case identifier
            vectors: List of vector dicts with 'vector_id', 'embedding', etc.
            
        Returns:
            Number of vectors stored
        """
        count = 0
        for vector_data in vectors:
            self.store_vector(
                case_id=case_id,
                vector_id=vector_data["vector_id"],
                embedding=vector_data["embedding"],
                entity_name=vector_data.get("entity_name"),
                metadata=vector_data.get("metadata")
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
        """Delete all data for a case."""
        query = "SELECT c.id FROM c WHERE c.case_id = @case_id"
        
        items = self.container.query_items(
            query=query,
            parameters=[{"name": "@case_id", "value": case_id}],
            partition_key=self._normalize_caseid(case_id),
            enable_cross_partition_query=False
        )
        
        for item in items:
            try:
                self.container.delete_item(item=item["id"], partition_key=case_id)
            except CosmosResourceNotFoundError:
                pass
