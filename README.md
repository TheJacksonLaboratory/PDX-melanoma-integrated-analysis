# PDX-melanoma-integrated-analysis

PDX melanoma 10x Visium Spatial Gene Expression integrated analysis

This repository contains source code and analysis notebooks for interated analysis of samples from WM4237 and WM4007 PDX models collected in a longitudinal experimental setup.


Module "Merging samples":
+ Load AnnData objects
+ Scale median total count per sample per spot
+ Evaluate library size distributions
+ Concatenate samples
+ Check correlation of total counts across samples (pre-filtering)

Module "Detect imaging artifacts":
+ Identify spots with imaging artifacts

Module "Detect mouse stroma"
+ Identify spots with relatively high mouse gene expression

Module "Filtering spots":
+ Remove imaging artifacts spots
+ Remove mouse stroma spots
+ Check correlation of total counts across samples (post-filtering)

Module "Dimensionality reduction":
+ Identify highly variable genes
+ Principal component analysis
+ Harmonize samples
+ Cluster spots
+ Generate 2D embeddings: UMAP, tSNE, PHATE

Module "DEG and overrepresentation analysis":
+ ...

Module "Gene sets scoring":
+ ...

Module "GSEA, ssGSEA":
+ ...

Module "Pathway plots":
+ Violin plots
+ Heatmap plots
+ UMAP plots
+ Spatial plots

Module "UMAP and Clustering on gene subsets":
+ ...


Module "RNA velocity":
+ Load velocyto reads quantification
+ Normalize total reads per spot (required for scVelo analysis)
+ Run scVelo analysis on combined samples
+ Run scVelo analysis on each sample independently

> "scVelo analysis" includes: velocity calculation, velocity on UMAP and spatial layout, visualization of velocity of specific genes, velocity graph, velocity pseudotime, PAGA.

Module "CNV analysis":
+ CaSpER-derived CNV analysis
+ InferCNV-derived CNV analysis
+ Continuous ordering

Module "Imaging features analysis":
+ Inception features dimensionality reduction and clustering
+ HoVer-Net-derived morphometrics analysis

Module "RNA continuous ordering analysis":
+ ...

Module "Fish plots of longitudinal data":
+ Visualization of clustering across time points (RNA, Imaging, CNV)
+ Visualization of clustering in continuous ordering (optional)
