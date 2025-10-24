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
        
        # Input container - stores input documents
        # Using /caseid as partition key for multi-tenant isolation
        # Each case's documents are in their own logical partition
        # caseid format: "12345.txt" for numeric cases, "ABC123" for alphanumeric
        print("[INFO] Creating 'input' container...")
        self.input_container = self.database.create_container_if_not_exists(
            id="input",
            partition_key=PartitionKey(path="/caseid")
        )
        print("[SUCCESS] Input container ready")
        
        # Output container - stores graph data AND vectors
        print("[INFO] Creating 'output' container...")
        self.output_container = self.database.create_container_if_not_exists(
            id="output",
            partition_key=PartitionKey(path="/caseid")
        )
        print("[SUCCESS] Output container ready")
        
        # Cache container - stores LLM cache
        print("[INFO] Creating 'cache' container...")
        self.cache_container = self.database.create_container_if_not_exists(
            id="cache",
            partition_key=PartitionKey(path="/caseid")
        )
        print("[SUCCESS] Cache container ready")
        
        # Logs container - stores processing logs
        print("[INFO] Creating 'logs' container...")
        self.logs_container = self.database.create_container_if_not_exists(
            id="logs",
            partition_key=PartitionKey(path="/caseid")
        )
        print("[SUCCESS] Logs container ready")
        
        # Config container - stores settings.yaml and prompts (for cloud deployment)
        # Config uses "global" as caseid since it's shared across all cases
        print("[INFO] Creating 'config' container...")
        self.config_container = self.database.create_container_if_not_exists(
            id="config",
            partition_key=PartitionKey(path="/case_id")  # Config still uses /case_id
        )
        print("[SUCCESS] Config container ready")
        
        # Vector store containers - for GraphRAG embeddings
        print("[INFO] Creating vector store containers...")
        
        # Entity description embeddings
        self.database.create_container_if_not_exists(
            id="output-entity-description",
            partition_key=PartitionKey(path="/id")
        )
        print("[SUCCESS] Entity description vector container ready")
        
        # Community full content embeddings  
        self.database.create_container_if_not_exists(
            id="output-community-full_content",
            partition_key=PartitionKey(path="/id")
        )
        print("[SUCCESS] Community content vector container ready")
        
        # Text unit embeddings
        self.database.create_container_if_not_exists(
            id="output-text_unit-text",
            partition_key=PartitionKey(path="/id")
        )
        print("[SUCCESS] Text unit vector container ready")
        
        print("[SUCCESS] All containers initialized successfully")


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
