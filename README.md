# Exploring Disease Similarity Through miRNA Profiles and Jaccard Index

In this analysis, we investigate disease-disease relationships based on Jaccard Index similarity, a metric commonly used to quantify the overlap between two sets. Here, the sets are defined by shared microRNAs (miRNAs) associated with each disease.

We construct a disease similarity matrix, where each cell represents the Jaccard similarity between a pair of diseases. To explore and interpret these relationships, we apply a combination of dimensionality reduction, clustering, and network analysis techniques:
- Dimensionality Reduction: We use t-SNE to project the high-dimensional similarity space into two dimensions.
- Clustering Algorithms: Techniques such as K-Means, Birch, Hierarchical Clustering, Gaussian Mixture Models, Mean Shift, and Affinity Propagation help uncover clusters of related diseases.
- Network Theory: We build an interactive similarity network where nodes represent diseases and edges denote similarity scores.

This tool enables you to:
- Perform clustering of diseases using multiple algorithms
- Interactively explore heatmaps and identify top related diseases
- Visualise miRNA-based disease similarity networks
