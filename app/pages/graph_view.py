import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from typing import Dict, Any, Optional

from app.components.ui_components import section_header, page_header, custom_divider, info_box


def render_graph_view():
    page_header("Interactive Graph Visualization")

    st.markdown(
        """
        <div class="info-box">
            Visualize the transaction graph where <b>nodes</b> represent customers and <b>edges</b> represent transactions.
            Green nodes = Normal activity, Red nodes = Fraud detected.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        n_nodes = st.slider("Number of Nodes", 10, 200, 80, key="graph_nodes")
    with col2:
        edge_density = st.slider("Edge Density", 0.1, 1.0, 0.4, key="graph_density")
    with col3:
        search_node = st.text_input("🔍 Search Node", placeholder="Enter node ID...")

    custom_divider()

    G = nx.erdos_renyi_graph(n=n_nodes, p=edge_density / n_nodes * 3, seed=42)
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    fraud_nodes = set(np.random.choice(G.nodes(), size=int(n_nodes * 0.12), replace=False))
    node_colors = ["#DC2626" if n in fraud_nodes else "#16A34A" for n in G.nodes()]
    node_sizes = [20 if n in fraud_nodes else 12 for n in G.nodes()]

    edge_trace_x, edge_trace_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_trace_x.extend([x0, x1, None])
        edge_trace_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_trace_x, y=edge_trace_y,
        line=dict(width=0.5, color="#CBD5E1"),
        hoverinfo="none",
        mode="lines",
    )

    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode="markers+text",
        text=[f"N{n}" for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=8, color="#64748B"),
        hoverinfo="text",
        hovertext=[
            f"<b>Node {n}</b><br>"
            f"Type: {'Fraud' if n in fraud_nodes else 'Normal'}<br>"
            f"Transactions: {G.degree(n)}<br>"
            f"Risk: {'High' if n in fraud_nodes else 'Low'}"
            for n in G.nodes()
        ],
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(color="white", width=1),
        ),
    )

    fraud_cluster_x = [pos[n][0] for n in fraud_nodes]
    fraud_cluster_y = [pos[n][1] for n in fraud_nodes]

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Transaction Graph Network",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        template="plotly_white",
        dragmode="pan",
    )

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="🔄 Reset View", method="relayout", args=[{"xaxis.range": None, "yaxis.range": None}]),
                    dict(label="🔍 Zoom to Fraud", method="relayout",
                         args=[{"xaxis.range": [min(fraud_cluster_x) - 0.5, max(fraud_cluster_x) + 0.5],
                                "yaxis.range": [min(fraud_cluster_y) - 0.5, max(fraud_cluster_y) + 0.5]}]),
                ],
                x=0.5,
                y=1.15,
                xanchor="center",
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)

    custom_divider()

    col1, col2 = st.columns(2)
    with col1:
        section_header("Graph Statistics")
        st.markdown(f"- **Total Nodes:** {G.number_of_nodes()}")
        st.markdown(f"- **Total Edges:** {G.number_of_edges()}")
        st.markdown(f"- **Fraud Nodes:** {len(fraud_nodes)} ({len(fraud_nodes)/n_nodes*100:.1f}%)")
        st.markdown(f"- **Avg Degree:** {sum(dict(G.degree()).values())/G.number_of_nodes():.2f}")
        st.markdown(f"- **Graph Density:** {nx.density(G):.4f}")

    with col2:
        section_header("Fraud Cluster Analysis")
        fraud_edges = sum(1 for u, v in G.edges() if u in fraud_nodes or v in fraud_nodes)
        normal_edges = G.number_of_edges() - fraud_edges
        st.markdown(f"- **Fraud-connected Edges:** {fraud_edges}")
        st.markdown(f"- **Normal Edges:** {normal_edges}")
        st.markdown(f"- **Fraud Edge Ratio:** {fraud_edges/G.number_of_edges()*100:.1f}%")
        st.markdown(f"- **Clustering Coefficient:** {nx.average_clustering(G):.4f}")

    custom_divider()

    section_header("Filter Graph")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_normal = st.checkbox("Show Normal Nodes", value=True)
    with col2:
        show_edges = st.checkbox("Show Edges", value=True)
    with col3:
        highlight_only = st.checkbox("Highlight Fraud Clusters Only", value=False)

    if highlight_only:
        info_box("🔍 Showing only fraud clusters. Isolating suspicious transaction patterns.", "warning")
