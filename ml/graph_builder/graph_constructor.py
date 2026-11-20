import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
import networkx as nx
from typing import Tuple, Dict, Any, Optional, List
from sklearn.neighbors import NearestNeighbors
from utils.logger import logger
from utils.config import MAX_GRAPH_NODES


class GraphConstructor:
    def __init__(self, max_nodes: int = MAX_GRAPH_NODES):
        self.max_nodes = max_nodes
        self.node_mapping: Dict[str, int] = {}
        self.reverse_mapping: Dict[int, str] = {}

    def build_graph(self, X: pd.DataFrame, y: pd.Series) -> Data:
        logger.info(f"Building graph with max {self.max_nodes} nodes")
        df = X.copy()
        df["label"] = y.values

        if len(df) > self.max_nodes:
            fraud_indices = df[df["label"] == 1].index
            normal_indices = df[df["label"] == 0].index
            n_fraud = len(fraud_indices)
            n_normal = min(self.max_nodes - n_fraud, len(normal_indices))
            if n_fraud + n_normal > self.max_nodes:
                n_normal = self.max_nodes - n_fraud
            selected_normal = np.random.choice(normal_indices, n_normal, replace=False)
            selected_indices = list(fraud_indices) + list(selected_normal)
            df = df.loc[selected_indices]

        num_nodes = len(df)
        node_features = torch.tensor(
            df.drop(columns=["label"]).select_dtypes(include=[np.number]).values,
            dtype=torch.float,
        )
        labels = torch.tensor(df["label"].values, dtype=torch.long)

        edge_index, edge_attr = self._build_edges(df, node_features)

        graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)
        graph_data.num_nodes = num_nodes
        logger.info(f"Graph built: {num_nodes} nodes, {edge_index.shape[1]} edges")
        return graph_data

    def _build_edges(self, df: pd.DataFrame, node_features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_nodes = len(df)
        knn = NearestNeighbors(n_neighbors=min(15, num_nodes - 1), metric="cosine")
        numeric_df = df.drop(columns=["label"]).select_dtypes(include=[np.number])
        if numeric_df.empty:
            return torch.zeros((2, 0), dtype=torch.long), None
        knn.fit(numeric_df.values)
        distances, indices = knn.kneighbors(numeric_df.values)

        edge_list = []
        edge_attr_list = []
        amount_col = None
        for col in df.columns:
            if "Amt" in col or "TransactionAmt" in col:
                amount_col = col
                break

        for i in range(num_nodes):
            for j, dist in zip(indices[i][1:], distances[i][1:]):
                if i != j:
                    edge_list.append([i, j])
                    attr = [1.0 / (1.0 + dist)]
                    if amount_col and amount_col in df.columns:
                        attr.append(float(df.iloc[i][amount_col]))
                    else:
                        attr.append(0.5)
                    edge_attr_list.append(attr)

        if not edge_list:
            return torch.zeros((2, 0), dtype=torch.long), None

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        return edge_index, edge_attr

    def build_networkx_graph(self, graph_data: Data) -> nx.Graph:
        G = nx.Graph()
        num_nodes = graph_data.num_nodes if hasattr(graph_data, "num_nodes") else graph_data.x.shape[0]
        for i in range(num_nodes):
            G.add_node(i, features=graph_data.x[i].tolist(), fraud=bool(graph_data.y[i].item()))

        edge_index = graph_data.edge_index
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            attr = graph_data.edge_attr[i].tolist() if graph_data.edge_attr is not None else None
            G.add_edge(src, dst, weight=attr[0] if attr else 1.0)

        return G

    def get_networkx_data(self, graph_data: Data, node_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        G = self.build_networkx_graph(graph_data)
        pos = nx.spring_layout(G, k=0.1, iterations=50)

        nodes = []
        for node in G.nodes():
            node_info = {
                "id": int(node),
                "x": float(pos[node][0]),
                "y": float(pos[node][1]),
                "fraud": bool(graph_data.y[node].item()),
                "features": graph_data.x[node].tolist()[:5],
            }
            if node_df is not None and node < len(node_df):
                node_info["label"] = str(node_df.iloc[node].name) if hasattr(node_df, 'iloc') else str(node)
            nodes.append(node_info)

        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                "source": int(u),
                "target": int(v),
                "weight": data.get("weight", 1.0),
            })

        return {"nodes": nodes, "edges": edges}
