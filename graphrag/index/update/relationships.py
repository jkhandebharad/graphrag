# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Relationship related operations and utils for Incremental Indexing."""

import itertools

import numpy as np
import pandas as pd

from graphrag.data_model.schemas import RELATIONSHIPS_FINAL_COLUMNS


def _update_and_merge_relationships(
    old_relationships: pd.DataFrame, delta_relationships: pd.DataFrame
) -> pd.DataFrame:
    """Update and merge relationships.

    Parameters
    ----------
    old_relationships : pd.DataFrame
        The old relationships.
    delta_relationships : pd.DataFrame
        The delta relationships.

    Returns
    -------
    pd.DataFrame
        The updated relationships.
    """
    # Increment the human readable id in b by the max of a
    # Delta relationships come with human_readable_id starting from 0 (from finalize_relationships)
    # We need to reassign them to avoid conflicts with old_relationships
    
    # First, ensure old_relationships has valid human_readable_id (handle NaN/missing)
    if "human_readable_id" not in old_relationships.columns:
        old_relationships["human_readable_id"] = old_relationships.index.astype(int)
    else:
        # Fill NaN with index-based values, then convert to int
        # Use mask to fill NaN values with corresponding index values
        mask = old_relationships["human_readable_id"].isna()
        if mask.any():
            old_relationships.loc[mask, "human_readable_id"] = old_relationships.index[mask]
        old_relationships["human_readable_id"] = old_relationships["human_readable_id"].astype(int)

    # Calculate the starting ID for delta relationships (must be > max of old)
    if len(old_relationships) == 0:
        initial_id = 0
    else:
        max_old_id = old_relationships["human_readable_id"].max()
        initial_id = int(max_old_id) + 1 if pd.notna(max_old_id) else 0

    # ALWAYS reassign delta_relationships human_readable_id starting from initial_id
    # This ensures no conflicts with existing IDs in old_relationships
    delta_relationships["human_readable_id"] = np.arange(
        initial_id, initial_id + len(delta_relationships)
    )

    # Merge the DataFrames without copying if possible
    merged_relationships = pd.concat(
        [old_relationships, delta_relationships], ignore_index=True, copy=False
    )

    # Group by title and resolve conflicts
    aggregated = (
        merged_relationships.groupby(["source", "target"])
        .agg({
            "id": "first",
            "human_readable_id": "first",
            "description": lambda x: list(x.astype(str)),  # Ensure str
            # Concatenate nd.array into a single list
            "text_unit_ids": lambda x: list(itertools.chain(*x.tolist())),
            "weight": "mean",
            "combined_degree": "sum",
        })
        .reset_index()
    )

    # Force the result into a DataFrame
    final_relationships: pd.DataFrame = pd.DataFrame(aggregated)

    # Recalculate target and source degrees
    final_relationships["source_degree"] = final_relationships.groupby("source")[
        "target"
    ].transform("count")
    final_relationships["target_degree"] = final_relationships.groupby("target")[
        "source"
    ].transform("count")

    # Recalculate the combined_degree of the relationships (source degree + target degree)
    final_relationships["combined_degree"] = (
        final_relationships["source_degree"] + final_relationships["target_degree"]
    )

    return final_relationships.loc[
        :,
        RELATIONSHIPS_FINAL_COLUMNS,
    ]
