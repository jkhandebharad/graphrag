# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing create_graph definition."""

import networkx as nx
import pandas as pd


def compute_degree(graph: nx.Graph) -> pd.DataFrame:
    """Create a new DataFrame with the degree of each node in the graph."""
    if len(graph) == 0:
        # Return empty DataFrame with correct columns if graph has no nodes
        return pd.DataFrame(columns=["title", "degree"])
    return pd.DataFrame([
        {"title": node, "degree": int(degree)}
        for node, degree in graph.degree  # type: ignore
    ])
