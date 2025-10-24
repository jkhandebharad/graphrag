"""
CosmosDB Logs Container Manager
Handles storage and retrieval of processing logs by case_id
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from app.cosmos_client import get_cosmos_manager
import uuid


class LogsManager:
    """Manages processing logs in CosmosDB."""
    
    def __init__(self):
        self.manager = None  # Will be set by get_firm_managers()
        self.container = None  # Will be set by get_firm_managers()
    
    def _normalize_caseid(self, case_id: str) -> str:
        """Normalize case_id for partition key (numeric cases get .txt extension)."""
        return f"{case_id}.txt" if case_id.isdigit() else case_id
    
    def log(
        self,
        case_id: str,
        level: str,
        message: str,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a log entry.
        
        Args:
            case_id: The case identifier
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            message: Log message
            source: Source of the log (e.g., 'indexing', 'query')
            metadata: Additional metadata
            
        Returns:
            The stored log document
        """
        log_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        # CosmosDB IDs cannot contain / : # ? or \ characters
        caseid = self._normalize_caseid(case_id)
        doc = {
            "id": f"logs-{case_id}-{timestamp.strftime('%Y%m%d%H%M%S')}-{log_id}",
            "caseid": caseid,  # Partition key
            "case_id": case_id,  # Original for queries/metadata
            "log_id": log_id,
            "timestamp": timestamp.isoformat(),
            "level": level,
            "message": message,
            "source": source or "system",
            "metadata": metadata or {}
        }
        
        return self.container.create_item(doc)
    
    def info(self, case_id: str, message: str, source: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log INFO level message."""
        return self.log(case_id, "INFO", message, source, metadata)
    
    def warning(self, case_id: str, message: str, source: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log WARNING level message."""
        return self.log(case_id, "WARNING", message, source, metadata)
    
    def error(self, case_id: str, message: str, source: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log ERROR level message."""
        return self.log(case_id, "ERROR", message, source, metadata)
    
    def debug(self, case_id: str, message: str, source: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log DEBUG level message."""
        return self.log(case_id, "DEBUG", message, source, metadata)
    
    def get_logs(
        self,
        case_id: str,
        level: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve logs for a case.
        
        Args:
            case_id: The case identifier
            level: Optional filter by log level
            source: Optional filter by source
            limit: Maximum number of logs to return
            
        Returns:
            List of log documents
        """
        query = "SELECT * FROM c WHERE c.case_id = @case_id"
        params = [{"name": "@case_id", "value": case_id}]
        
        if level:
            query += " AND c.level = @level"
            params.append({"name": "@level", "value": level})
        
        if source:
            query += " AND c.source = @source"
            params.append({"name": "@source", "value": source})
        
        query += " ORDER BY c.timestamp DESC"
        
        items = self.container.query_items(
            query=query,
            parameters=params,
            partition_key=self._normalize_caseid(case_id),
            enable_cross_partition_query=False,
            max_item_count=limit
        )
        
        return list(items)
    
    def get_recent_logs(self, case_id: str, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get logs from the last N minutes."""
        from datetime import timedelta
        cutoff_time = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
        
        query = """
        SELECT * FROM c 
        WHERE c.case_id = @case_id 
        AND c.timestamp >= @cutoff_time
        ORDER BY c.timestamp DESC
        """
        
        items = self.container.query_items(
            query=query,
            parameters=[
                {"name": "@case_id", "value": case_id},
                {"name": "@cutoff_time", "value": cutoff_time}
            ],
            partition_key=self._normalize_caseid(case_id),
            enable_cross_partition_query=False
        )
        
        return list(items)
    
    def clear_logs(self, case_id: str):
        """Delete all logs for a case."""
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
    
    def store_batch_logs(self, case_id: str, log_content: str, source: str = "batch"):
        """
        Store a batch of logs (e.g., from a log file).
        
        Args:
            case_id: The case identifier
            log_content: The log file content
            source: Source identifier
        """
        log_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        # CosmosDB IDs cannot contain / : # ? or \ characters
        doc = {
            "id": f"logs-{case_id}-batch-{timestamp.strftime('%Y%m%d%H%M%S')}-{log_id}",
            "case_id": case_id,
            "log_id": log_id,
            "timestamp": timestamp.isoformat(),
            "level": "INFO",
            "message": "Batch log upload",
            "source": source,
            "content": log_content,
            "metadata": {"type": "batch"}
        }
        
        return self.container.create_item(doc)
