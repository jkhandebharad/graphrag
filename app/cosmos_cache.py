"""
CosmosDB Cache Container Manager
Handles storage and retrieval of LLM cache entries by case_id
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from app.cosmos_client import get_cosmos_manager
import hashlib


class CacheManager:
    """Manages LLM cache in CosmosDB."""
    
    def __init__(self):
        self.manager = None  # Will be set by get_firm_managers()
        self.container = None  # Will be set by get_firm_managers()
    
    # No longer needed - using /id as partition key in case-specific containers
    
    def store_cache_entry(
        self,
        case_id: str,
        cache_key: str,
        cache_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a cache entry.
        
        Args:
            case_id: The case identifier
            cache_key: Cache key (e.g., hash from GraphRAG)
            cache_type: Type of cache (e.g., 'text_embedding', 'chat_completion')
            content: The cached content
            metadata: Additional metadata
            
        Returns:
            The stored cache document
        """
        # CosmosDB IDs cannot contain / : # ? or \ characters
        doc = {
            "id": f"cache-{case_id}-{cache_type}-{cache_key}",
            "case_id": case_id,  # Keep for metadata/queries
            "cache_key": cache_key,
            "cache_type": cache_type,
            "content": content,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        return self.container.upsert_item(doc)
    
    def get_cache_entry(
        self,
        case_id: str,
        cache_type: str,
        cache_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cache entry.
        
        Args:
            case_id: The case identifier
            cache_type: Type of cache
            cache_key: Cache key
            
        Returns:
            The cache document or None if not found
        """
        try:
            doc_id = f"cache-{case_id}-{cache_type}-{cache_key}"
            return self.container.read_item(item=doc_id, partition_key=doc_id)
        except CosmosResourceNotFoundError:
            return None
    
    def list_cache_entries(
        self,
        case_id: str,
        cache_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List cache entries for a case.
        
        Args:
            case_id: The case identifier
            cache_type: Optional filter by cache type
            
        Returns:
            List of cache documents
        """
        if cache_type:
            query = """
            SELECT * FROM c 
            WHERE c.case_id = @case_id 
            AND c.cache_type = @cache_type
            """
            params = [
                {"name": "@case_id", "value": case_id},
                {"name": "@cache_type", "value": cache_type}
            ]
        else:
            query = "SELECT * FROM c WHERE c.case_id = @case_id"
            params = [{"name": "@case_id", "value": case_id}]
        
        items = self.container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=False
        )
        
        return list(items)
    
    def delete_cache_entry(self, case_id: str, cache_type: str, cache_key: str):
        """Delete a specific cache entry."""
        try:
            doc_id = f"cache-{case_id}-{cache_type}-{cache_key}"
            self.container.delete_item(item=doc_id, partition_key=doc_id)
        except CosmosResourceNotFoundError:
            pass
    
    def clear_cache(self, case_id: str, cache_type: Optional[str] = None):
        """
        Clear cache entries for a case.
        
        Args:
            case_id: The case identifier
            cache_type: Optional - if provided, only clear entries of this type
        """
        entries = self.list_cache_entries(case_id, cache_type)
        for entry in entries:
            try:
                self.container.delete_item(item=entry["id"], partition_key=entry["id"])
            except CosmosResourceNotFoundError:
                pass
    
    def get_cache_stats(self, case_id: str) -> Dict[str, Any]:
        """Get statistics about cache usage for a case."""
        entries = self.list_cache_entries(case_id)
        
        stats = {
            "total_entries": len(entries),
            "by_type": {},
            "total_size_estimate": 0
        }
        
        for entry in entries:
            cache_type = entry.get("cache_type", "unknown")
            stats["by_type"][cache_type] = stats["by_type"].get(cache_type, 0) + 1
            
            # Rough size estimate (content as string)
            content_str = str(entry.get("content", ""))
            stats["total_size_estimate"] += len(content_str)
        
        return stats
