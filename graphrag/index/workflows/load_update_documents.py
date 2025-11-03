# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging

import pandas as pd

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.config.models.input_config import InputConfig
from graphrag.index.input.factory import create_input
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.update.incremental_index import get_delta_docs
from graphrag.storage.pipeline_storage import PipelineStorage
from graphrag.utils.storage import write_table_to_storage

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Load and parse update-only input documents into a standard format."""
    logger.info("[LOAD_UPDATE] run_workflow called - starting load_update_documents workflow")
    
    output = await load_update_documents(
        config.input,
        context.input_storage,
        context.previous_storage,
    )

    logger.info(f"[LOAD_UPDATE] Final # of update rows loaded: {len(output)}")
    context.stats.update_documents = len(output)

    if len(output) == 0:
        logger.warning("[LOAD_UPDATE] No new update documents found - stopping pipeline")
        return WorkflowFunctionOutput(result=None, stop=True)

    logger.info(f"[LOAD_UPDATE] Writing {len(output)} new documents to output storage")
    await write_table_to_storage(output, "documents", context.output_storage)

    return WorkflowFunctionOutput(result=output)


async def load_update_documents(
    config: InputConfig,
    input_storage: PipelineStorage,
    previous_storage: PipelineStorage,
) -> pd.DataFrame:
    """Load and parse update-only input documents into a standard format."""
    logger.info("[LOAD_UPDATE] load_update_documents workflow called - this is incremental indexing")
    prev_container = getattr(previous_storage, '_container_name', 'unknown')
    input_container = getattr(input_storage, '_container_name', 'unknown')
    logger.info(f"[LOAD_UPDATE] previous_storage container: {prev_container}")
    logger.info(f"[LOAD_UPDATE] input_storage container: {input_container}")
    
    input_documents = await create_input(config, input_storage)
    logger.info(f"[LOAD_UPDATE] Loaded {len(input_documents)} documents from input storage")
    
    # previous storage is the output of the previous run
    # we'll use this to diff the input from the prior
    logger.info("[LOAD_UPDATE] Calling get_delta_docs to compare with previous documents...")
    delta_documents = await get_delta_docs(input_documents, previous_storage)
    
    logger.info(f"[LOAD_UPDATE] Delta comparison complete: {len(delta_documents.new_inputs)} new documents, {len(delta_documents.deleted_inputs)} deleted documents")
    
    return delta_documents.new_inputs
