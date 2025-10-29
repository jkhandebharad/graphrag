"""
CosmosDB Client and Container Management for GraphRAG Multi-Case Storage
"""
import os
from typing import Optional, Dict, Any, List
from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceNotFoundError, CosmosHttpResponseError
from dotenv import load_dotenv

load_dotenv()

class CosmosDBManager:
    """Manages CosmosDB connections and container operations."""
    
    def __init__(self):
        # Load configuration from environment
        self.endpoint = os.getenv("COSMOS_ENDPOINT")
        self.key = os.getenv("COSMOS_KEY")
        self.database_name = os.getenv("COSMOS_DATABASE_NAME", "graphrag-db")
        
        if not self.endpoint or not self.key:
            raise ValueError("COSMOS_ENDPOINT and COSMOS_KEY must be set in environment variables")
        
        # Initialize client
        self.client = CosmosClient(self.endpoint, self.key)
        self.database = None
        
        # Container references
        self.input_container = None
        self.output_container = None
        self.cache_container = None
        self.logs_container = None
        self.config_container = None  # For settings.yaml and prompts
        
    def initialize_database(self):
        """Create database and containers if they don't exist."""
        print(f"[INFO] Initializing CosmosDB database: {self.database_name}")
        
        # Create database
        self.database = self.client.create_database_if_not_exists(id=self.database_name)
        print(f"[SUCCESS] Database ready: {self.database_name}")
        
        # Create containers with appropriate partition keys
        self._create_containers()
        
    def _create_containers(self):
        """Create all required containers with partition keys."""
        print("[INFO] Creating general containers (config only)...")
        
        # Config container - stores settings.yaml and prompts (for cloud deployment)
        # Config uses "global" as caseid since it's shared across all cases
        print("[INFO] Creating 'config' container...")
        self.config_container = self.database.create_container_if_not_exists(
            id="config",
            partition_key=PartitionKey(path="/case_id")  # Config still uses /case_id
        )
        print("[SUCCESS] Config container ready")
        
        print("[INFO] Case-specific containers will be created dynamically per case")
        print("[SUCCESS] General containers initialized successfully")
    
    def create_case_specific_containers(self, case_id: str):
        """Create case-specific containers for a given case_id."""
        print(f"[INFO] Creating case-specific containers for case: {case_id}")
        
        # Input container - stores input documents for this case
        input_container_name = f"input_{case_id}"
        print(f"[INFO] Creating '{input_container_name}' container...")
        self.input_container = self.database.create_container_if_not_exists(
            id=input_container_name,
            partition_key=PartitionKey(path="/id")
        )
        print(f"[SUCCESS] Input container '{input_container_name}' ready")
        
        # Output container - stores graph data AND vectors for this case
        output_container_name = f"output_{case_id}"
        print(f"[INFO] Creating '{output_container_name}' container...")
        self.output_container = self.database.create_container_if_not_exists(
            id=output_container_name,
            partition_key=PartitionKey(path="/id")
        )
        print(f"[SUCCESS] Output container '{output_container_name}' ready")
        
        # Cache container - stores LLM cache for this case
        cache_container_name = f"cache_{case_id}"
        print(f"[INFO] Creating '{cache_container_name}' container...")
        self.cache_container = self.database.create_container_if_not_exists(
            id=cache_container_name,
            partition_key=PartitionKey(path="/id")
        )
        print(f"[SUCCESS] Cache container '{cache_container_name}' ready")
        
        # Logs container - stores processing logs for this case
        logs_container_name = f"logs_{case_id}"
        print(f"[INFO] Creating '{logs_container_name}' container...")
        self.logs_container = self.database.create_container_if_not_exists(
            id=logs_container_name,
            partition_key=PartitionKey(path="/id")
        )
        print(f"[SUCCESS] Logs container '{logs_container_name}' ready")
        
        # Create vector store containers with case-specific names that match GraphRAG's expectations
        print(f"[INFO] Creating case-specific vector store containers for case: {case_id}")
        
        # Define only the 3 vector store containers that GraphRAG actually uses
        # GraphRAG calls create_index_name("output", "entity.description") -> "output_5678-entity-description"
        vector_containers = [
            f"output_{case_id}-entity-description",
            f"output_{case_id}-community-full_content",
            f"output_{case_id}-text_unit-text"
        ]
        
        for container_name in vector_containers:
            try:
                self.database.create_container_if_not_exists(
                    id=container_name,
                    partition_key=PartitionKey(path="/id")
                )
                print(f"[SUCCESS] Vector container '{container_name}' ready")
            except Exception as e:
                print(f"[WARNING] Failed to create vector container '{container_name}': {e}")
        
        print(f"[DEBUG] Created vector containers: {vector_containers}")
        
        print(f"[SUCCESS] All case-specific containers for case '{case_id}' initialized successfully")


# Global instance
_cosmos_manager: Optional[CosmosDBManager] = None

def get_cosmos_manager() -> CosmosDBManager:
    """Get or create the global CosmosDB manager instance."""
    global _cosmos_manager
    if _cosmos_manager is None:
        _cosmos_manager = CosmosDBManager()
        _cosmos_manager.initialize_database()
    return _cosmos_manager

def init_cosmos_db():
    """Initialize CosmosDB (call this at app startup)."""
    get_cosmos_manager()
