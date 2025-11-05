# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing create_graph definition."""

import networkx as nx
import pandas as pd


def compute_degree(graph: nx.Graph) -> pd.DataFrame:
    """Create a new DataFrame with the degree of each node in the graph.
    graph : nx.Graph
        NetworkX Graph object (undirected graph) containing nodes and edges.
        
        Example graph structure:
        - Nodes: ["Michael Anderson", "John Davis", "Brooklyn, New York"]
        - Edges:
          * ("Michael Anderson", "John Davis")
          * ("John Davis", "Brooklyn, New York")
     Returns
    -------
    pd.DataFrame
        DataFrame with node degrees containing columns:
        - title (str): Node identifier/name
        - degree (int): Number of edges connected to this node
        
        Example output:
        ┌─────────────────────┬────────┐
        │ title               │ degree │
        ├─────────────────────┼────────┤
        │ Michael Anderson    │ 1      │  # Connected to 1 node (John Davis)
        │ John Davis          │ 2      │  # Connected to 2 nodes (Michael Anderson, Brooklyn)
        │ Brooklyn, New York  │ 1      │  # Connected to 1 node (John Davis)
        └─────────────────────┴────────┘
    """
    if len(graph) == 0:
        # Return empty DataFrame with correct columns if graph has no nodes
        return pd.DataFrame(columns=["title", "degree"])
    return pd.DataFrame([
        {"title": node, "degree": int(degree)}
        for node, degree in graph.degree  # type: ignore
    ])
