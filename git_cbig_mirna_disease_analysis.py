import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn.mixture import GaussianMixture
from numpy import unique
import plotly.express as px
import plotly.colors as pc
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

file_id = "1qzP5GkOzbcJB4a0SxB09XFxdOGqMT20u"
csv_url = f"https://drive.google.com/uc?export=download&id={file_id}"

jcmat = pd.read_csv(csv_url, index_col=0)
# List of diseases in order from the similarity matrix
diseases = jcmat.index.tolist()

# Adding new mesh id and disease name table
mesh_file_id = "15M5Sa5fVG_BKP8ciy7U-qoks2cmNVil8"
mesh_csv_url = f"https://drive.google.com/uc?export=download&id={mesh_file_id}"

mapping_df = pd.read_csv(mesh_csv_url)
id_to_names = mapping_df.groupby("disease_mesh_id")["disease_mesh_name"].apply(lambda x: list(set(", ".join(x).split(", ")))).to_dict()
names_to_id = mapping_df.groupby("disease_mesh_name")["disease_mesh_id"].apply(lambda x: list(set(x))).to_dict()

def get_disease_label(mesh_id):
    names = id_to_names.get(mesh_id, ["Unknown"])
    return f"{mesh_id} — {', '.join(names)}"

# Convert similarity matrix to distance matrix for clustering (distance = 1 - similarity)
distance_matrix = 1 - jcmat.values

# Load mapping of MeSH IDs to names, change this into a dtive file
#mapping_path = r'C:\Users\vikir\Downloads\New_MESH_with_names.csv'  # The CSV you created
#mapping_df = pd.read_csv(mapping_path)
#id_to_names = mapping_df.groupby("disease_mesh_id")["disease_mesh_name"].apply(lambda x: list(set(", ".join(x).split(", ")))).to_dict()

# Use a remote file for production
mesh_file_id = "15M5Sa5fVG_BKP8ciy7U-qoks2cmNVil8"
mesh_csv_url = f"https://drive.google.com/uc?export=download&id={mesh_file_id}"
mapping_df = pd.read_csv(mesh_csv_url)

# Function to get display label for a MeSH ID or combined ID
def get_disease_label(mesh_id):
    names = id_to_names.get(mesh_id, ["Unknown"])
    return f"{mesh_id} — {', '.join(names)}"



# Create two main tabs for better organization: clustering view and network view
tab1, tab2 = st.tabs(["Clustering View", "Similarity Network"])

# --------------
# TAB 1: Clustering View
# --------------
with tab1:
    st.header("Clustering Settings")

    # User inputs to control number of clusters and clustering method
    n_clusters = st.slider("Number of Clusters", 2, 50, 10)
    clustering_method = st.selectbox("Clustering Method", [
        "KMeans", "Birch", "Hierarchical", "Gaussian Mixture", "MeanShift", "Affinity Propagation"
    ])
    min_cluster_size = st.slider("Minimum Cluster Size for Selection", 1, 100, 1)

    st.subheader("Clustering-based Visualization of Diseases")

    # Reduce dimensionality to 2D for visualization using t-SNE on the distance matrix
    tsne = TSNE(metric='precomputed', perplexity=30, random_state=42, init='random')
    X_embedded = tsne.fit_transform(distance_matrix)

    # ------------------------
    # Define clustering function
    # ------------------------
    def clusteringAlgorithms(X, method, n_clusters=30):
        try:
            if method == 'KMeans':
                mdl = KMeans(n_clusters=n_clusters)
                yNew = mdl.fit_predict(X)
            elif method == 'Birch':
                mdl = Birch(threshold=0.05, n_clusters=n_clusters)
                mdl.fit(X)
                yNew = mdl.predict(X)
            elif method == 'Hierarchical':
                mdl = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean')
                yNew = mdl.fit_predict(X)
            elif method == 'Gaussian Mixture':
                mdl = GaussianMixture(n_components=n_clusters)
                mdl.fit(X)
                yNew = mdl.predict(X)
            elif method == 'MeanShift':
                bandwidth = estimate_bandwidth(X, quantile=0.2)
                mdl = MeanShift(bandwidth=bandwidth)
                yNew = mdl.fit_predict(X)
            elif method == 'Affinity Propagation':
                mdl = AffinityPropagation(preference=-10, damping=0.9)
                mdl.fit(X)
                yNew = mdl.predict(X)
            else:
                raise ValueError("Invalid method")
            clusters = unique(yNew)
            return X, yNew, clusters, method
        except Exception as e:
            st.error(f"Clustering error: {e}")
            return X, np.zeros(len(X)), [], "Failed"

    # Run the chosen clustering algorithm on the 2D data
    X_, yNew, clusters, Alg = clusteringAlgorithms(X_embedded, clustering_method, n_clusters)

    # Build a DataFrame with cluster labels, coordinates, and disease names for plotting & filtering
    df = pd.DataFrame({
        "x": X_[:, 0],
        "y": X_[:, 1],
        "cluster": yNew.astype(str),
        "disease": diseases
    })

    df["label"] = df["disease"].apply(get_disease_label)

    color_sequence = pc.qualitative.Alphabet + pc.qualitative.Pastel + pc.qualitative.Set3
    fig = px.scatter(df, x="x", y="y", color="cluster", hover_name="label",
                     color_discrete_sequence=color_sequence, title=f"{Alg} Clustering", width=900, height=650)
    st.plotly_chart(fig, use_container_width=True)

    # UI row with 2 toggles
    col1, col2 = st.columns([1, 1])
    with col1:
        use_cluster_selection = st.checkbox("Select Disease by Cluster", value=False)
    with col2:
        selection_mode = st.radio("Select by:", ["MeSH ID", "Disease Name"], horizontal=True)

    # Determine options based on cluster
    cluster_sizes = df['cluster'].value_counts().to_dict()
    valid_clusters = [c for c, size in cluster_sizes.items() if size >= min_cluster_size]

    if use_cluster_selection:
        selected_cluster = st.selectbox("Select a Cluster", sorted(valid_clusters, key=int))
        options = df[df["cluster"] == selected_cluster]["disease"].tolist()
    else:
        options = diseases

    # Build dropdown labels depending on toggle
    if selection_mode == "MeSH ID":
        display_options = {d: get_disease_label(d).split(" — ")[0] for d in options}
    else:
        display_options = {d: ", ".join(id_to_names.get(d, ["Unknown"])) for d in options}

    reversed_display = {v: k for k, v in display_options.items()}
    selected_display = st.selectbox("Select a Disease", list(reversed_display.keys()))
    selected_disease = reversed_display[selected_display]

    top_n = st.slider("Top N Most Similar Diseases", 5, 50, 20)
    selected_cluster_label = df[df["disease"] == selected_disease]["cluster"].values[0]
    cluster_members = df[df["cluster"] == selected_cluster_label]["disease"].tolist()

    st.subheader(f"Cluster Members of Selected Disease `{get_disease_label(selected_disease)}`")
    cluster_similarities = jcmat.loc[selected_disease, cluster_members]
    st.selectbox("Cluster Members", [f"{get_disease_label(d)} (Similarity: {cluster_similarities[d]:.2f})" for d in cluster_similarities.index])

    st.subheader(f"Top {top_n} Similar Diseases to `{get_disease_label(selected_disease)}`")
    top_similar = jcmat.loc[selected_disease].drop(selected_disease).sort_values(ascending=False).head(top_n)
    st.selectbox("Similar Diseases", [f"{get_disease_label(d)} (Similarity: {top_similar[d]:.2f})" for d in top_similar.index])

    heatmap_data_2 = jcmat.loc[top_similar.index, top_similar.index]
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data_2, cmap="viridis", annot=False, xticklabels=True, yticklabels=True, ax=ax2)
    plt.xticks(rotation=90)
    st.pyplot(fig2)

    cluster_heatmap_data = jcmat.loc[cluster_members, cluster_members]
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(cluster_heatmap_data, cmap="viridis", annot=False, xticklabels=True, yticklabels=True, ax=ax3)
    plt.xticks(rotation=90)
    st.pyplot(fig3)

    selected_cluster_for_heatmap = st.selectbox("Select Cluster to Visualize / Show Heatmap", sorted(df["cluster"].unique(), key=int))
    heatmap_cluster_data = jcmat.loc[
        df[df["cluster"] == selected_cluster_for_heatmap]["disease"],
        df[df["cluster"] == selected_cluster_for_heatmap]["disease"]
    ]
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_cluster_data, cmap="viridis", annot=False, xticklabels=True, yticklabels=True, ax=ax4)
    plt.xticks(rotation=90)
    st.pyplot(fig4)

# --------------
# TAB 2: Disease Similarity Network
# --------------
with tab2:
    st.subheader("Disease Similarity Network")

    # Threshold slider for minimum similarity to draw an edge
    threshold = st.slider("Minimum Jaccard Similarity for Edge", 0.0, 1.0, 0.3, 0.01)

    # Allow user to select diseases to include in network
    all_labeled_diseases = [get_disease_label(d) for d in diseases]
    label_to_disease = {get_disease_label(d): d for d in diseases}

    selected_display_labels = st.multiselect("Select diseases to include", options=all_labeled_diseases)

    if not selected_display_labels:
        max_nodes = st.slider("No selection made. Showing top N most connected diseases", 10, min(len(jcmat), 100), 30)
        total_similarity = jcmat.sum(axis=1)
        top_diseases = total_similarity.sort_values(ascending=False).head(max_nodes).index
        df_subset = jcmat.loc[top_diseases, top_diseases]
    else:
        selected_ids = [label_to_disease[lbl] for lbl in selected_display_labels]
        df_subset = jcmat.loc[selected_ids, selected_ids]


    # Build network graph with NetworkX
    G = nx.Graph()
    for disease in df_subset.index:
        G.add_node(disease, label=get_disease_label(disease))

    for i, disease1 in enumerate(df_subset.index):
        for j, disease2 in enumerate(df_subset.columns):
            if i < j:
                weight = df_subset.loc[disease1, disease2]
                if weight >= threshold:
                    edge_width = (weight ** 3) * 900
                    G.add_edge(
                        disease1, disease2,
                        weight=weight,
                        title=f"Similarity: {weight:.2f}",
                        width=edge_width
                    )

    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)
    net.repulsion(node_distance=200, central_gravity=0.3)
    for node in net.nodes:
        node["title"] = node["label"]
        node["label"] = node["label"]

    net.save_graph("graph.html")
    HtmlFile = open("graph.html", "r", encoding="utf-8")
    components.html(HtmlFile.read(), height=750, scrolling=True)

# df = pd.read_csv(csv_url, index_col=0)

# st.title("Disease Similarity Network")

# # Threshold slider
# threshold = st.slider("Minimum Jaccard Similarity for Edge", 0.0, 1.0, 0.3, 0.01)

# # Multi-select box to choose diseases
# all_diseases = df.index.tolist()
# selected_diseases = st.multiselect(
#     "Select diseases to include in the network (leave blank for top N default)",
#     options=all_diseases
# )

# # Default to top N if none selected
# if not selected_diseases:
#     max_nodes = st.slider("No selection made. Showing top N most connected diseases", 10, min(len(df), 100), 30)

#     # Calculate total similarity per disease (sum of Jaccard scores)
#     total_similarity = df.sum(axis=1)
#     top_diseases = total_similarity.sort_values(ascending=False).head(max_nodes).index

#     df_subset = df.loc[top_diseases, top_diseases]
# else:
#     df_subset = df.loc[selected_diseases, selected_diseases]

# # Build graph
# G = nx.Graph()

# # Add nodes
# for disease in df_subset.index:
#     G.add_node(disease)

# # Add edges
# for i, disease1 in enumerate(df_subset.index):
#     for j, disease2 in enumerate(df_subset.columns):
#         if i < j:
#             weight = df_subset.loc[disease1, disease2]
#             if weight >= threshold:
#                 edge_width = (weight ** 3) * 900  # Cubic scaling
#                 G.add_edge(
#                     disease1,
#                     disease2,
#                     weight=weight,
#                     title=f"Similarity: {weight:.2f}",
#                     width=edge_width
#                 )

# net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
# net.from_nx(G)
# net.repulsion(node_distance=200, central_gravity=0.3)

# net.save_graph("graph.html")
# HtmlFile = open("graph.html", "r", encoding="utf-8")
# components.html(HtmlFile.read(), height=750, scrolling=True)
