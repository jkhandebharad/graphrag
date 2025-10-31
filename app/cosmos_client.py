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
    
    def check_case_containers_exist(self, case_id: str) -> bool:
        """
        Check if case-specific containers already exist AND have indexed data.
        Incremental indexing should only happen if output container has final entities/relationships.
        
        Args:
            case_id: Case identifier to check
            
        Returns:
            True if containers exist AND output container has indexed data (final entities), False otherwise
        """
        if not self.database:
            return False
        
        required_containers = [
            f"input_{case_id}",
            f"output_{case_id}",
            f"cache_{case_id}",
            f"logs_{case_id}"
        ]
        
        # First, check if all containers exist
        for container_name in required_containers:
            try:
                container = self.database.get_container_client(container_name)
                # Try to read container properties to verify it exists
                container.read()
            except CosmosResourceNotFoundError:
                # Container doesn't exist - this is a first run
                print(f"[INFO] Container '{container_name}' does not exist - this is a first run (full indexing)")
                return False
            except Exception as e:
                print(f"[WARNING] Error checking container '{container_name}': {e}")
                return False
        
        # All containers exist, now check if output container has actual indexed data
        # Check for final entities (with UUID IDs) - these indicate completed indexing
        # Final entities have UUIDs like "entities:51d4a181-c2e7-46f9-9896-81c784dbacec"
        # Raw entities have numeric IDs like "entities:0", "entities:1"
        output_container_name = f"output_{case_id}"
        try:
            output_container = self.database.get_container_client(output_container_name)
            
            # Query for entities - check if any have UUID pattern (contain hyphens)
            # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (has 4 hyphens)
            query = "SELECT TOP 10 c.id FROM c WHERE STARTSWITH(c.id, 'entities:')"
            results = list(output_container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            
            if not results:
                # No entities at all - this is a first run
                print(f"[INFO] Output container has no entities - this is a first run (full indexing)")
                return False
            
            # Check if any entity has UUID pattern (contains hyphens after 'entities:')
            # UUID IDs are much longer and contain hyphens
            has_final_entities = False
            for item in results:
                entity_id = item.get("id", "")
                if ":" in entity_id:
                    id_part = entity_id.split(":", 1)[1]
                    # Check if it's a UUID (contains hyphens) or just numeric
                    # UUIDs contain hyphens, numeric IDs don't
                    if "-" in id_part and len(id_part) > 20:  # UUIDs are long and have hyphens
                        has_final_entities = True
                        break
            
            if has_final_entities:
                # Found final entities - indexing has been completed before
                print(f"[INFO] Output container has indexed data (final entities with UUIDs found) - incremental indexing will be used")
                return True
            else:
                # Only raw entities (numeric IDs) or no entities - this is a first run
                print(f"[INFO] Output container has no final entities (only raw/numeric IDs) - this is a first run (full indexing)")
                return False
                
        except Exception as e:
            print(f"[WARNING] Error checking for indexed data in output container: {e}")
            # If we can't check, assume it's a first run to be safe
            return False
    
    def create_update_output_container(self, case_id: str):
        """
        Create the update_output container for incremental indexing.
        This container stores delta and previous data during incremental updates.
        
        Args:
            case_id: Case identifier for container naming
        """
        if not self.database:
            raise ValueError("Database not initialized")
        
        update_container_name = f"update_output_{case_id}"
        print(f"[INFO] Creating update output container '{update_container_name}'...")
        
        try:
            self.database.create_container_if_not_exists(
                id=update_container_name,
                partition_key=PartitionKey(path="/id")
            )
            print(f"[SUCCESS] Update output container '{update_container_name}' ready")
        except Exception as e:
            print(f"[ERROR] Failed to create update output container: {e}")
            raise


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
