import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

file_id = "1qzP5GkOzbcJB4a0SxB09XFxdOGqMT20u"
csv_url = f"https://drive.google.com/uc?export=download&id={file_id}"

df = pd.read_csv(csv_url, index_col=0)

st.title("Disease Similarity Network")

# Threshold slider
threshold = st.slider("Minimum Jaccard Similarity for Edge", 0.0, 1.0, 0.3, 0.01)

# Multi-select box to choose diseases
all_diseases = df.index.tolist()
selected_diseases = st.multiselect(
    "Select diseases to include in the network (leave blank for top N default)",
    options=all_diseases
)

# Default to top N if none selected
if not selected_diseases:
    max_nodes = st.slider("No selection made. Showing top N most connected diseases", 10, min(len(df), 100), 30)

    # Calculate total similarity per disease (sum of Jaccard scores)
    total_similarity = df.sum(axis=1)
    top_diseases = total_similarity.sort_values(ascending=False).head(max_nodes).index

    df_subset = df.loc[top_diseases, top_diseases]
else:
    df_subset = df.loc[selected_diseases, selected_diseases]

# Build graph
G = nx.Graph()

# Add nodes
for disease in df_subset.index:
    G.add_node(disease)

# Add edges
for i, disease1 in enumerate(df_subset.index):
    for j, disease2 in enumerate(df_subset.columns):
        if i < j:
            weight = df_subset.loc[disease1, disease2]
            if weight >= threshold:
                edge_width = (weight ** 3) * 900  # Cubic scaling
                G.add_edge(
                    disease1,
                    disease2,
                    weight=weight,
                    title=f"Similarity: {weight:.2f}",
                    width=edge_width
                )

net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
net.from_nx(G)
net.repulsion(node_distance=200, central_gravity=0.3)

net.save_graph("graph.html")
HtmlFile = open("graph.html", "r", encoding="utf-8")
components.html(HtmlFile.read(), height=750, scrolling=True)
