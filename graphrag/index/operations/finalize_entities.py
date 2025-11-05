# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""All the steps to transform final entities."""

from uuid import uuid4

import pandas as pd

from graphrag.config.models.embed_graph_config import EmbedGraphConfig
from graphrag.data_model.schemas import ENTITIES_FINAL_COLUMNS
from graphrag.index.operations.compute_degree import compute_degree
from graphrag.index.operations.create_graph import create_graph
from graphrag.index.operations.embed_graph.embed_graph import embed_graph
from graphrag.index.operations.layout_graph.layout_graph import layout_graph


def finalize_entities(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    embed_config: EmbedGraphConfig | None = None,
    layout_enabled: bool = False,
) -> pd.DataFrame:
    """All the steps to transform final entities.
     Input examples:
    
    entities DataFrame:
    ┌─────────────────────┬──────────┬──────────────────────────────┬──────────────┬──────────┐
    │ title               │ type     │ description                  │ text_unit_ids│ frequency│
    ├─────────────────────┼──────────┼──────────────────────────────┼──────────────┼──────────┤
    │ Michael Anderson    │ person   │ Plaintiff in accident case   │ [tu-1, tu-2] │ 2        │
    │ John Davis          │ person   │ Driver of Ford Explorer      │ [tu-1]       │ 1        │
    │ Brooklyn, New York  │ geo      │ Location of the accident     │ [tu-1]       │ 1        │
    └─────────────────────┴──────────┴──────────────────────────────┴──────────────┴──────────┘
    
    relationships DataFrame:
    ┌─────────────────────┬─────────────────────┬──────────────┬────────┬──────────────┐
    │ source              │ target               │ description  │ weight │ text_unit_ids│
    ├─────────────────────┼─────────────────────┼──────────────┼────────┼──────────────┤
    │ Michael Anderson    │ John Davis           │ was struck by│ 1.0    │ [tu-1]       │
    │ John Davis          │ Michael Anderson     │ struck       │ 1.0    │ [tu-1]       │
    └─────────────────────┴─────────────────────┴──────────────┴────────┴──────────────┘
    """
    graph = create_graph(relationships, edge_attr=["weight"])
    graph_embeddings = None
    if embed_config is not None and embed_config.enabled:
        graph_embeddings = embed_graph(
            graph,
            embed_config,
        )
    layout = layout_graph(
        graph,
        layout_enabled,
        embeddings=graph_embeddings,
    )
    degrees = compute_degree(graph)
    final_entities = (
        entities.merge(layout, left_on="title", right_on="label", how="left")
        .merge(degrees, on="title", how="left")
        .drop_duplicates(subset="title")
    )
    final_entities = final_entities.loc[entities["title"].notna()].reset_index()
    # disconnected nodes and those with no community even at level 0 can be missing degree
    # Also handle case where graph is empty and degree column wasn't created by merge
    if "degree" not in final_entities.columns:
        final_entities["degree"] = 0
    else:
        final_entities["degree"] = final_entities["degree"].fillna(0).astype(int)
    # Handle missing layout columns (x, y) when graph is empty or merge fails
    if "x" not in final_entities.columns:
        final_entities["x"] = 0.0
    else:
        final_entities["x"] = final_entities["x"].fillna(0.0).astype(float)
    if "y" not in final_entities.columns:
        final_entities["y"] = 0.0
    else:
        final_entities["y"] = final_entities["y"].fillna(0.0).astype(float)
    final_entities.reset_index(inplace=True)
    final_entities["human_readable_id"] = final_entities.index
    final_entities["id"] = final_entities["human_readable_id"].apply(
        lambda _x: str(uuid4())
    )

    '''
    Returns
    ┌──────────────────────────────────────┬───────────────────┬─────────────────────┬──────────┬──────────────────────────────┬──────────────┬──────────┬────────┬──────┬──────┐
    │ id                                   │ human_readable_id │ title               │ type     │ description                  │ text_unit_ids│ frequency│ degree │ x    │ y    │
    ├──────────────────────────────────────┼───────────────────┼─────────────────────┼──────────┼──────────────────────────────┼──────────────┼──────────┼────────┼──────┼──────┤
    │ 51d4a181-c2e7-46f9-9896-81c784dbacec │ 0                  │ Michael Anderson    │ person   │ Plaintiff in accident case   │ [tu-1, tu-2] │ 2        │ 1      │ 0.0  │ 0.0  │
    │ fc83687e-3476-4db6-982e-79f0dd4c8ccd │ 1                  │ John Davis          │ person   │ Driver of Ford Explorer      │ [tu-1]       │ 1        │ 2      │ 0.0  │ 0.0  │
    │ 47453779-0abe-41ae-813c-e2f245a79a1e │ 2                  │ Brooklyn, New York  │ geo      │ Location of the accident     │ [tu-1]       │ 1        │ 1      │ 0.0  │ 0.0  │
    └──────────────────────────────────────┴───────────────────┴─────────────────────┴──────────┴──────────────────────────────┴──────────────┴──────────┴────────┴──────┴──────┘
    '''
    return final_entities.loc[
        :,
        ENTITIES_FINAL_COLUMNS,
    ]
