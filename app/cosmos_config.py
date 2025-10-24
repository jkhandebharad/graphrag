"""
CosmosDB Configuration Container Manager
Handles storage of settings.yaml and prompts for cloud deployment
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from app.cosmos_client import get_cosmos_manager


class ConfigManager:
    """Manages configuration files (settings.yaml, prompts) in CosmosDB."""
    
    def __init__(self):
        # Use local files for config and prompts - no CosmosDB needed
        self.manager = None
        self.container = None
    
    def store_settings_yaml(self, content: str) -> Dict[str, Any]:
        """
        Store settings.yaml in CosmosDB.
        
        Args:
            content: The YAML content as string
            
        Returns:
            The stored document
        """
        doc = {
            "id": "settings-yaml",
            "case_id": "global",  # For partition key
            "type": "settings",
            "content": content,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        return self.container.upsert_item(doc)
    
    def get_settings_yaml(self) -> Optional[str]:
        """Retrieve settings.yaml content from local file."""
        import os
        settings_path = os.path.join("ragtest", "settings.yaml")
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return None
    
    def store_prompt(self, prompt_name: str, content: str) -> Dict[str, Any]:
        """
        Store a prompt file in CosmosDB.
        
        Args:
            prompt_name: Name of the prompt (e.g., "extract_graph.txt")
            content: The prompt content
            
        Returns:
            The stored document
        """
        doc = {
            "id": f"prompt-{prompt_name.replace('.', '-')}",
            "case_id": "global",
            "type": "prompt",
            "prompt_name": prompt_name,
            "content": content,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        return self.container.upsert_item(doc)
    
    def get_prompt(self, prompt_name: str) -> Optional[str]:
        """Retrieve a prompt file content from CosmosDB."""
        try:
            doc_id = f"prompt-{prompt_name.replace('.', '-')}"
            doc = self.container.read_item(item=doc_id, partition_key="global")
            return doc.get("content")
        except CosmosResourceNotFoundError:
            return None
    
    def list_prompts(self) -> List[str]:
        """List all stored prompts."""
        query = """
        SELECT c.prompt_name 
        FROM c 
        WHERE c.type = 'prompt'
        """
        
        items = self.container.query_items(
            query=query,
            partition_key="global",
            enable_cross_partition_query=False
        )
        
        return [item["prompt_name"] for item in items]
    
    def get_all_prompts(self) -> Dict[str, str]:
        """Get all prompts as a dictionary {filename: content} from local files."""
        import os
        import glob
        
        prompts = {}
        prompts_dir = os.path.join("ragtest", "prompts")
        
        if os.path.exists(prompts_dir):
            # Read all .txt files from prompts directory
            for prompt_file in glob.glob(os.path.join(prompts_dir, "*.txt")):
                filename = os.path.basename(prompt_file)
                try:
                    with open(prompt_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        prompts[filename] = content
                except Exception as e:
                    print(f"[WARNING] Could not read prompt file {filename}: {e}")
        
        return prompts


# Global instance
config_manager = ConfigManager()

