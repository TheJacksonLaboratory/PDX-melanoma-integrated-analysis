# PDX-melanoma-integrated-analysis

PDX melanoma 10x Visium Spatial Gene Expression integrated analysis

This repository contains source code and analysis notebooks for interated analysis of samples from WM4237 and WM4007 PDX models collected in a longitudinal experimental setup.


Module "Merging samples":
+ [x] Load AnnData objects
+ [x] Scale median total count per sample per spot
+ [x] Evaluate library size distributions
+ [x] Concatenate samples
+ [x] Check correlation of total counts across samples (pre-filtering)

Module "Detect imaging artifacts":
+ [x] Identify spots with imaging artifacts

Module "Detect mouse stroma"
+ [x] Identify spots with high mouse gene expression

Module "Filtering spots":
+ [x] Remove mouse stroma spots
+ [x] Check correlation of total counts across samples (post-filtering)

Module "Dimensionality reduction":
+ [x] Identify highly variable genes
+ [x] Principal component analysis
+ [x] Harmonize samples
+ [x] Cluster spots
+ [x] Generate 2D embeddings: UMAP, tSNE, PHATE

Module "CNV analysis":
+ [x] CaSpER-derived CNV analysis
+ [x] InferCNV-derived CNV analysis
+ [x] Continuous ordering

Module "Imaging features analysis, integration with ST WSI":
+ [x] Inception features dimensionality reduction and clustering
+ [x] Integration from ST H&Es and non-ST H&Es
+ [x] HoVer-Net-derived morphometrics overview

Module "DEG and overrepresentation analysis":
+ [x] Untreated & Controls: all vs all clusters
+ [x] Treated: all vs all clusters
+ [x] Early vs late persisters
+ [x] Early sensitive vs late resistant

Module "Pathway plots":
+ [x] Gene sets scores plots
+ [x] Heatmap plots
+ [x] Spatial plots
+ [ ] UMAP plots

Module "Fish plots of longitudinal data":
+ [x] Visualization of clustering across time points (RNA, Imaging, CNV)
+ [x] Visualization of clustering in continuous ordering (optional)

Module "UMAP and Clustering on gene subsets":
+ [ ] MAPK
+ [ ] ECM

Module "RNA velocity":
+ [ ] Load velocyto reads quantification
+ [ ] Normalize total reads per spot (required for scVelo analysis)
+ [ ] Run scVelo analysis on combined samples
+ [ ] Run scVelo analysis on each sample independently

> "scVelo analysis" includes: velocity calculation, velocity on UMAP and spatial layout, visualization of velocity of specific genes, velocity graph, velocity pseudotime, PAGA.


> On hold:
> 
> Module "Gene Set Enrichment Analysis":
> + [ ] GSEA
> + [ ] ssGSEA


> On hold:
> 
> Module "RNA continuous ordering analysis":
> + [ ] ...
