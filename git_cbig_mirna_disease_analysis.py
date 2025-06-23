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

# Convert similarity matrix to distance matrix for clustering (distance = 1 - similarity)
distance_matrix = 1 - jcmat.values

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
    tsne = TSNE(metric='precomputed', perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(distance_matrix)

    # ------------------------
    # Define clustering function
    # ------------------------
    def clusteringAlgorithms(X, chooseClustering, n_clusters=30):
        Alg = ''
        yNew = None
        try:
            if chooseClustering == 'KMeans':
                mdl = KMeans(n_clusters=n_clusters)
                yNew = mdl.fit_predict(X)
                Alg = 'KMEANS'

            elif chooseClustering == 'Birch':
                mdl = Birch(threshold=0.05, n_clusters=n_clusters)
                mdl.fit(X)
                yNew = mdl.predict(X)
                Alg = 'BIRCH'

            elif chooseClustering == 'Hierarchical':
                mdl = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean')
                yNew = mdl.fit_predict(X)
                Alg = 'HIERARCHICAL'

            elif chooseClustering == 'Gaussian Mixture':
                mdl = GaussianMixture(n_components=n_clusters)
                mdl.fit(X)
                yNew = mdl.predict(X)
                Alg = 'GAUSSIAN MIXTURE'

            elif chooseClustering == 'MeanShift':
                bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=X.shape[0])
                mdl = MeanShift(bandwidth=bandwidth)
                yNew = mdl.fit_predict(X)
                Alg = 'MEANSHIFT'

            elif chooseClustering == 'Affinity Propagation':
                mdl = AffinityPropagation(preference=-10, damping=0.9)
                mdl.fit(X)
                yNew = mdl.predict(X)
                Alg = 'AFFINITY PROPAGATION'

        except Exception as e:
            st.error(f"Error with {Alg}: {e}")
            return X, np.zeros(len(X)), [], f"{Alg} (Failed)"

        clusters = unique(yNew)
        return X, yNew, clusters, Alg

    # Run the chosen clustering algorithm on the 2D data
    X_, yNew, clusters, Alg = clusteringAlgorithms(X_embedded, clustering_method, n_clusters)

    # Build a DataFrame with cluster labels, coordinates, and disease names for plotting & filtering
    df = pd.DataFrame({
        "x": X_[:, 0],
        "y": X_[:, 1],
        "cluster": yNew.astype(str),  # cast to string for categorical color mapping
        "disease": diseases
    })

    # ------------------------
    # Interactive clustering scatter plot using Plotly
    # ------------------------
    color_sequence = pc.qualitative.Alphabet + pc.qualitative.Pastel + pc.qualitative.Set3
    fig = px.scatter(
        df, x="x", y="y", color="cluster", hover_data=["disease"],
        color_discrete_sequence=color_sequence,
        title=f"{Alg} Clustering (Click legend items to show/hide clusters)",
        width=900, height=650
    )
    fig.update_layout(legend_title="Cluster", legend_itemsizing='constant')
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------
    # Disease selection: either by cluster or global list
    # ------------------------
    use_cluster_selection = st.checkbox("Select Disease by Cluster", value=False)
    cluster_sizes = df['cluster'].value_counts().to_dict()
    # Only consider clusters with enough members
    valid_clusters = [c for c, size in cluster_sizes.items() if size >= min_cluster_size]
    all_diseases = diseases.copy()

    if use_cluster_selection:
        selected_cluster_for_disease = st.selectbox("Select a Cluster for Disease Selection", sorted(valid_clusters, key=lambda x: int(x)))
        # Diseases in selected cluster
        cluster_diseases = df[df["cluster"] == selected_cluster_for_disease]["disease"].tolist()
        selected_disease = st.selectbox("Select Disease from Cluster", cluster_diseases)
    else:
        selected_disease = st.selectbox("Select a Disease", all_diseases)

    # Number of top similar diseases to show
    top_n = st.slider("Top N Most Similar Diseases", min_value=5, max_value=50, value=20)

    # Find the cluster label of the selected disease
    selected_cluster_label = df[df["disease"] == selected_disease]["cluster"].values[0]
    # Get all diseases in that cluster
    cluster_members = df[df["cluster"] == selected_cluster_label]["disease"].tolist()

    # ------------------------
    # Show cluster members in an expander with similarity values
    # ------------------------
    st.subheader(f"Cluster Members of Selected Disease `{selected_disease}` (Cluster {selected_cluster_label})")
    similarities_cluster = jcmat.loc[selected_disease, cluster_members]

    with st.expander("Show/Hide Cluster Members"):
        for d in similarities_cluster.index:
            # Show disease and its similarity to selected disease
            st.markdown(f"- {d} (Similarity: {similarities_cluster[d]:.2f})")

    # ------------------------
    # Show top N similar diseases with similarity values inside expander
    # ------------------------
    st.subheader(f"Top {top_n} Similar Diseases to `{selected_disease}`")
    # Sort descending excluding the selected disease itself
    top_similar = jcmat.loc[selected_disease].drop(selected_disease).sort_values(ascending=False).head(top_n)

    with st.expander("Show/Hide Top Similar Diseases"):
        for d in top_similar.index:
            st.markdown(f"- {d} (Similarity: {top_similar[d]:.2f})")

    # ------------------------
    # Heatmap for top similar diseases' internal similarities
    # ------------------------
    st.subheader(f"Heatmap: Internal Similarities Among Top {top_n} Similar Diseases")
    heatmap_data_2 = jcmat.loc[top_similar.index, top_similar.index]

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data_2, cmap="viridis", annot=False, xticklabels=True, yticklabels=True, ax=ax2)
    plt.xticks(rotation=90)
    st.pyplot(fig2)

    # ------------------------
    # Heatmap for the cluster members
    # ------------------------
    st.subheader(f"Heatmap: Similarities Within Selected Cluster (Cluster {selected_cluster_label})")
    cluster_heatmap_data = jcmat.loc[cluster_members, cluster_members]

    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(cluster_heatmap_data, cmap="viridis", annot=False, xticklabels=True, yticklabels=True, ax=ax3)
    plt.xticks(rotation=90)
    st.pyplot(fig3)

    # ------------------------
    # User can select another cluster to visualize its heatmap independently
    # ------------------------
    selected_cluster_for_heatmap = st.selectbox(
        "Select Cluster to Visualize / Show Heatmap",
        sorted(df["cluster"].unique(), key=lambda x: int(x))
    )

    st.subheader(f"Heatmap: Similarities Within Cluster {selected_cluster_for_heatmap} (Independent Selection)")
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
    selected_diseases = st.multiselect(
        "Select diseases to include in the network (leave blank for top N default)",
        options=diseases
    )

    if not selected_diseases:
        # If none selected, show top N most connected diseases by total similarity sum
        max_nodes = st.slider("No selection made. Showing top N most connected diseases", 10, min(len(jcmat), 100), 30)
        total_similarity = jcmat.sum(axis=1)
        top_diseases = total_similarity.sort_values(ascending=False).head(max_nodes).index
        df_subset = jcmat.loc[top_diseases, top_diseases]
    else:
        # Subset similarity matrix for selected diseases
        df_subset = jcmat.loc[selected_diseases, selected_diseases]

    # Build network graph with NetworkX
    G = nx.Graph()
    for disease in df_subset.index:
        G.add_node(disease)

    # Add edges with weight above threshold
    for i, disease1 in enumerate(df_subset.index):
        for j, disease2 in enumerate(df_subset.columns):
            if i < j:
                weight = df_subset.loc[disease1, disease2]
                if weight >= threshold:
                    # Edge width scales cubically for better visual difference
                    edge_width = (weight ** 3) * 900
                    G.add_edge(
                        disease1,
                        disease2,
                        weight=weight,
                        title=f"Similarity: {weight:.2f}",
                        width=edge_width
                    )

    # Visualize with pyvis network, rendered inside Streamlit
    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)
    net.repulsion(node_distance=200, central_gravity=0.3)

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
