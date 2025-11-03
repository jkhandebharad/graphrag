# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Encapsulates pipeline construction and selection."""

import logging
from typing import ClassVar

from graphrag.config.enums import IndexingMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.typing.pipeline import Pipeline
from graphrag.index.typing.workflow import WorkflowFunction

logger = logging.getLogger(__name__)


class PipelineFactory:
    """A factory class for workflow pipelines."""

    workflows: ClassVar[dict[str, WorkflowFunction]] = {}
    pipelines: ClassVar[dict[str, list[str]]] = {}

    @classmethod
    def register(cls, name: str, workflow: WorkflowFunction):
        """Register a custom workflow function."""
        cls.workflows[name] = workflow

    @classmethod
    def register_all(cls, workflows: dict[str, WorkflowFunction]):
        """Register a dict of custom workflow functions."""
        for name, workflow in workflows.items():
            cls.register(name, workflow)

    @classmethod
    def register_pipeline(cls, name: str, workflows: list[str]):
        """Register a new pipeline method as a list of workflow names."""
        cls.pipelines[name] = workflows

    @classmethod
    def create_pipeline(
        cls,
        config: GraphRagConfig,
        method: IndexingMethod | str = IndexingMethod.Standard,
    ) -> Pipeline:
        """Create a pipeline generator."""
        original_method = method
        logger.debug(f"[FACTORY] create_pipeline called with method: {method} (type: {type(method).__name__})")
        
        # Convert string method to enum if needed (for incremental indexing)
        if isinstance(method, str):
            logger.debug(f"[FACTORY] Method is string: '{method}', attempting to convert to IndexingMethod enum...")
            try:
                method = IndexingMethod(method)
                logger.info(f"[FACTORY] Successfully converted string '{original_method}' to enum: {method} (value: '{method.value}')")
            except ValueError as e:
                logger.error(f"[FACTORY] Failed to convert string '{method}' to IndexingMethod enum: {e}")
                logger.error(f"[FACTORY] Available enum values: {[e.value for e in IndexingMethod]}")
                # If the string doesn't match an enum value, use it directly
                logger.warning(f"[FACTORY] Using string '{method}' directly for lookup")
        
        logger.debug(f"[FACTORY] Looking up pipeline with method: {method} (type: {type(method).__name__})")
        logger.debug(f"[FACTORY] Available pipeline keys: {[str(k) for k in cls.pipelines.keys()]}")
        logger.debug(f"[FACTORY] Pipeline keys types: {[type(k).__name__ for k in cls.pipelines.keys()]}")
        
        # Check if we're doing incremental indexing
        is_update_method = (
            method == IndexingMethod.StandardUpdate or 
            method == IndexingMethod.FastUpdate or
            (isinstance(method, str) and method.endswith("-update"))
        )
        
        # Handle both cases: workflows set in config OR method-based lookup
        if config.workflows is not None and len(config.workflows) > 0:
            # CASE 1: Workflows explicitly set in settings.yaml
            logger.info(f"[FACTORY] workflows explicitly set in config ({len(config.workflows)} workflows)")
            
            if is_update_method:
                # Incremental indexing detected - adjust workflows automatically
                logger.info("[FACTORY] Incremental indexing detected - adjusting workflows from config")
                workflows = list(config.workflows)  # Make a copy to avoid modifying the original
                
                # Replace load_input_documents with load_update_documents if present
                if "load_input_documents" in workflows:
                    idx = workflows.index("load_input_documents")
                    workflows[idx] = "load_update_documents"
                    logger.info("[FACTORY] Replaced 'load_input_documents' with 'load_update_documents'")
                elif "load_update_documents" not in workflows:
                    # If load_update_documents is missing, add it at the beginning
                    workflows.insert(0, "load_update_documents")
                    logger.info("[FACTORY] Added 'load_update_documents' at the start")
                
                # Add update workflows if not already present
                # These must come after the standard workflows
                update_workflows = [
                    "update_final_documents",
                    "update_entities_relationships",
                    "update_text_units",
                    "update_covariates",
                    "update_communities",
                    "update_community_reports",
                    "update_text_embeddings",
                    "update_clean_state",
                ]
                
                # Find the position after generate_text_embeddings (last standard workflow)
                # or append at the end if generate_text_embeddings not found
                if "generate_text_embeddings" in workflows:
                    insert_pos = workflows.index("generate_text_embeddings") + 1
                else:
                    insert_pos = len(workflows)
                
                # Add update workflows that aren't already in the list
                for update_wf in update_workflows:
                    if update_wf not in workflows:
                        workflows.insert(insert_pos, update_wf)
                        insert_pos += 1
                        logger.debug(f"[FACTORY] Added update workflow: {update_wf}")
                
                logger.info(f"[FACTORY] Adjusted workflows for incremental indexing ({len(workflows)} total): {workflows}")
            else:
                # Standard indexing - use workflows as-is
                workflows = config.workflows
                logger.info(f"[FACTORY] Using workflows from config as-is (standard indexing)")
        else:
            # CASE 2: No workflows in config - use method-based pipeline lookup
            logger.info(f"[FACTORY] No workflows in config, using method-based pipeline lookup")
            workflows = cls.pipelines.get(method, [])
            
            if not workflows:
                logger.error(f"[FACTORY] No pipeline found for method '{method}'! Available methods: {list(cls.pipelines.keys())}")
                logger.warning(f"[FACTORY] Falling back to empty list - this may cause issues")
        
        logger.info(f"[FACTORY] Creating pipeline with {len(workflows)} workflows: {workflows}")
        if workflows:
            logger.info(f"[FACTORY] First workflow in pipeline: {workflows[0] if len(workflows) > 0 else 'NONE'}")
        
        return Pipeline([(name, cls.workflows[name]) for name in workflows])


# --- Register default implementations ---
_standard_workflows = [
    "create_base_text_units",
    "create_final_documents",
    "extract_graph",
    "finalize_graph",
    "extract_covariates",
    "create_communities",
    "create_final_text_units",
    "create_community_reports",
    "generate_text_embeddings",
]
_fast_workflows = [
    "create_base_text_units",
    "create_final_documents",
    "extract_graph_nlp",
    "prune_graph",
    "finalize_graph",
    "create_communities",
    "create_final_text_units",
    "create_community_reports_text",
    "generate_text_embeddings",
]
_update_workflows = [
    "update_final_documents",
    "update_entities_relationships",
    "update_text_units",
    "update_covariates",
    "update_communities",
    "update_community_reports",
    "update_text_embeddings",
    "update_clean_state",
]
PipelineFactory.register_pipeline(
    IndexingMethod.Standard, ["load_input_documents", *_standard_workflows]
)
PipelineFactory.register_pipeline(
    IndexingMethod.Fast, ["load_input_documents", *_fast_workflows]
)
PipelineFactory.register_pipeline(
    IndexingMethod.StandardUpdate,
    ["load_update_documents", *_standard_workflows, *_update_workflows],
)
PipelineFactory.register_pipeline(
    IndexingMethod.FastUpdate,
    ["load_update_documents", *_fast_workflows, *_update_workflows],
)
