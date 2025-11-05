# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""All the steps to transform final relationships."""

from uuid import uuid4

import pandas as pd

from graphrag.data_model.schemas import RELATIONSHIPS_FINAL_COLUMNS
from graphrag.index.operations.compute_degree import compute_degree
from graphrag.index.operations.compute_edge_combined_degree import (
    compute_edge_combined_degree,
)
from graphrag.index.operations.create_graph import create_graph


def finalize_relationships(
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    """All the steps to transform final relationships.
    
    Input examples:
    
    relationships DataFrame:
    ┌─────────────────────┬─────────────────────┬──────────────┬────────┬──────────────┐
    │ source              │ target               │ description  │ weight │ text_unit_ids│
    ├─────────────────────┼─────────────────────┼──────────────┼────────┼──────────────┤
    │ Michael Anderson    │ John Davis           │ was struck by│ 1.0    │ [tu-1]       │
    │ John Davis          │ Michael Anderson     │ struck       │ 1.0    │ [tu-1]       │
    └─────────────────────┴─────────────────────┴──────────────┴────────┴──────────────┘
    """
    graph = create_graph(relationships, edge_attr=["weight"])
    degrees = compute_degree(graph)

    final_relationships = relationships.drop_duplicates(subset=["source", "target"])
    final_relationships["combined_degree"] = compute_edge_combined_degree(
        final_relationships,
        degrees,
        node_name_column="title",
        node_degree_column="degree",
        edge_source_column="source",
        edge_target_column="target",
    )

    final_relationships.reset_index(inplace=True)
    final_relationships["human_readable_id"] = final_relationships.index
    final_relationships["id"] = final_relationships["human_readable_id"].apply(
        lambda _x: str(uuid4())
    )
    '''
    Returns
    ┌─────────────────────┬─────────────────────┬──────────────┬────────┬──────────────┬───────────────────┬───────────────────┬────────┐
    │ id                  │ source              │ target               │ description  │ weight │ text_unit_ids│ human_readable_id │ combined_degree │
    ├─────────────────────┼─────────────────────┼─────────────────────┼──────────────┼────────┼──────────────┼───────────────────┼───────────────────┤
    │ 51d4a181-c2e7-46f9-9896-81c784dbacec │ Michael Anderson    │ John Davis           │ was struck by│ 1.0    │ [tu-1]       │ 0                 │ 2               │
    │ fc83687e-3476-4db6-982e-79f0dd4c8ccd │ John Davis          │ Michael Anderson     │ struck       │ 1.0    │ [tu-1]       │ 1                 │ 2               │
    └─────────────────────┴─────────────────────┴─────────────────────┴──────────────┴────────┴──────────────┴───────────────────┴───────────────────┘
    '''
    return final_relationships.loc[
        :,
        RELATIONSHIPS_FINAL_COLUMNS,
    ]
