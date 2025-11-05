# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing create_graph definition."""

import networkx as nx
import pandas as pd


def create_graph(
    edges: pd.DataFrame,
    edge_attr: list[str | int] | None = None,
    nodes: pd.DataFrame | None = None,
    node_id: str = "title",
) -> nx.Graph:
    """Create a networkx graph from nodes and edges dataframes.
    
    Input examples:
    
    edges DataFrame:
    ┌─────────────────────┬─────────────────────┬──────────────┬────────┬──────────────┐
    │ source              │ target               │ description  │ weight │ text_unit_ids│
    ├─────────────────────┼─────────────────────┼──────────────┼────────┼──────────────┤
    │ Michael Anderson    │ John Davis           │ was struck by│ 1.0    │ [tu-1]       │
    │ John Davis          │ Michael Anderson     │ struck       │ 1.0    │ [tu-1]       │
    └─────────────────────┴─────────────────────┴──────────────┴────────┴──────────────┘
  
    nodes DataFrame:
    ┌─────────────────────┬──────────┬──────────────────────────────┬──────────────┬──────────┐
    │ title               │ type     │ description                  │ text_unit_ids│ frequency│
    ├─────────────────────┼──────────┼──────────────────────────────┼──────────────┼──────────┤
    │ Michael Anderson    │ person   │ Plaintiff in accident case   │ [tu-1, tu-2] │ 2        │
    │ John Davis          │ person   │ Driver of Ford Explorer      │ [tu-1]       │ 1        │
    │ Brooklyn, New York  │ geo      │ Location of the accident     │ [tu-1]       │ 1        │
    └─────────────────────┴──────────┴──────────────────────────────┴──────────────┴──────────┘
    """
    graph = nx.from_pandas_edgelist(edges, edge_attr=edge_attr)

    if nodes is not None:
        nodes.set_index(node_id, inplace=True)
        graph.add_nodes_from((n, dict(d)) for n, d in nodes.iterrows())
    '''
    Returns
    -------
    nx.Graph
        NetworkX Graph object (undirected graph) with:
        - Nodes: All unique values from source and target columns in edges
        - Edges: Connections between source and target nodes
        - Edge attributes: Columns specified in edge_attr (e.g., weight, text_unit_ids)
        - Node attributes: All columns from nodes DataFrame (e.g., type, description, frequency, text_unit_ids)
        
        Example output structure:
        - Nodes: ["Michael Anderson", "John Davis"]
        - Edges: [("Michael Anderson", "John Davis")]
        - Edge attributes: {"weight": 1.0, "text_unit_ids": ["tu-1"]}
        - Node attributes: {"type": "person", "frequency": 2, "text_unit_ids": ["tu-1", "tu-2"], ...}
    '''
    return graph
