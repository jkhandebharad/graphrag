# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Dataframe operations and utils for Incremental Indexing."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from graphrag.storage.pipeline_storage import PipelineStorage
from graphrag.utils.storage import (
    load_table_from_storage,
    write_table_to_storage,
)


@dataclass
class InputDelta:
    """Dataclass to hold the input delta.

    Attributes
    ----------
    new_inputs : pd.DataFrame
        The new inputs.
    deleted_inputs : pd.DataFrame
        The deleted inputs.
    """

    new_inputs: pd.DataFrame
    deleted_inputs: pd.DataFrame


async def get_delta_docs(
    input_dataset: pd.DataFrame, storage: PipelineStorage
) -> InputDelta:
    """Get the delta between the input dataset and the final documents.

    Parameters
    ----------
    input_dataset : pd.DataFrame
        The input dataset.
    storage : PipelineStorage
        The Pipeline storage.

    Returns
    -------
    InputDelta
        The input delta. With new inputs and deleted inputs.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"[DELTA] Starting document comparison - input dataset has {len(input_dataset)} documents")
    logger.debug(f"[DELTA] Input dataset columns: {input_dataset.columns.tolist()}")
    if "title" in input_dataset.columns:
        logger.debug(f"[DELTA] Input dataset titles: {input_dataset['title'].tolist()}")
    else:
        logger.error(f"[DELTA] Input dataset missing 'title' column!")
    
    try:
        final_docs = await load_table_from_storage("documents", storage)
        logger.info(f"[DELTA] Loaded {len(final_docs)} documents from previous storage")
        logger.debug(f"[DELTA] Previous storage columns: {final_docs.columns.tolist() if not final_docs.empty else 'EMPTY'}")
        
        if final_docs.empty:
            logger.warning("[DELTA] Previous documents DataFrame is EMPTY - all input documents will be treated as NEW")
            return InputDelta(
                new_inputs=input_dataset.copy(),
                deleted_inputs=pd.DataFrame()
            )
        
        if "title" not in final_docs.columns:
            logger.error(f"[DELTA] Previous documents missing 'title' column! Available: {final_docs.columns.tolist()}")
            return InputDelta(
                new_inputs=input_dataset.copy(),
                deleted_inputs=pd.DataFrame()
            )
        
        if "title" not in input_dataset.columns:
            logger.error(f"[DELTA] Input dataset missing 'title' column! Available: {input_dataset.columns.tolist()}")
            return InputDelta(
                new_inputs=input_dataset.copy(),
                deleted_inputs=pd.DataFrame()
            )
    except ValueError as e:
        logger.error(f"[DELTA] Failed to load documents from previous storage: {e}")
        logger.warning("[DELTA] Treating all input documents as NEW")
        return InputDelta(
            new_inputs=input_dataset.copy(),
            deleted_inputs=pd.DataFrame()
        )

    # Select distinct title from final docs and from dataset
    previous_docs: list[str] = final_docs["title"].unique().tolist()
    dataset_docs: list[str] = input_dataset["title"].unique().tolist()
    
    logger.info(f"[DELTA] Previous documents titles ({len(previous_docs)}): {previous_docs}")
    logger.info(f"[DELTA] Input documents titles ({len(dataset_docs)}): {dataset_docs}")

    # Get the new documents (using loc to ensure DataFrame)
    new_docs = input_dataset.loc[~input_dataset["title"].isin(previous_docs)]

    # Get the deleted documents (again using loc to ensure DataFrame)
    deleted_docs = final_docs.loc[~final_docs["title"].isin(dataset_docs)]
    
    logger.info(f"[DELTA] Comparison result: {len(new_docs)} NEW documents, {len(deleted_docs)} DELETED documents")
    if len(new_docs) > 0:
        logger.info(f"[DELTA] NEW document titles: {new_docs['title'].tolist()}")
    if len(deleted_docs) > 0:
        logger.info(f"[DELTA] DELETED document titles: {deleted_docs['title'].tolist()}")

    return InputDelta(new_docs, deleted_docs)


async def concat_dataframes(
    name: str,
    previous_storage: PipelineStorage,
    delta_storage: PipelineStorage,
    output_storage: PipelineStorage,
) -> pd.DataFrame:
    """Concatenate dataframes."""
    old_df = await load_table_from_storage(name, previous_storage)
    delta_df = await load_table_from_storage(name, delta_storage)

    # Merge the final documents
    initial_id = old_df["human_readable_id"].max() + 1
    delta_df["human_readable_id"] = np.arange(initial_id, initial_id + len(delta_df))
    final_df = pd.concat([old_df, delta_df], ignore_index=True, copy=False)

    await write_table_to_storage(final_df, name, output_storage)

    return final_df
