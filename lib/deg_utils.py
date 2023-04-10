
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()
importr('Seurat')

import pandas as pd
import numpy as np


def DEG(adata, account_batch_var=False, identity='', batchvar='sample', subsample=1000, logfc_threshold=0.15, saveMarkers=False):
    print('adata shape:', adata.shape)
    data = adata.X.todense().T
    robjects.r.assign('data', robjects.r.matrix(data, nrow=data.shape[0], ncol=data.shape[1]))
    robjects.r.assign('genes', robjects.StrVector(adata.var.index.values))
    robjects.r.assign('spots', robjects.StrVector(adata.obs.index.values))
    robjects.r('row.names(data) <- genes')
    robjects.r('colnames(data) <- spots')
    robjects.r('dim(data)')
    robjects.r('se <- Seurat::CreateSeuratObject(counts=as(data, "sparseMatrix"))');

    robjects.r.assign('clusters_all', robjects.StrVector(adata.obs[identity].astype(str).values))
    robjects.r('Idents(object=se) <- clusters_all');
    
    robjects.r.assign('sample', robjects.StrVector(adata.obs[batchvar].astype(str).values))
    robjects.r('names(sample) <- colnames(x=se)')
    robjects.r('se <- AddMetaData(object=se, metadata=sample, col.name="sample")')
    
    if account_batch_var:
        print("Correcting for batch variable:", batchvar)
        robjects.r('cluster_markers_all <- Seurat::FindAllMarkers(object=se, max.cells.per.ident=%s, assay=NULL, logfc.threshold=%s, min.pct=0.05, slot="counts", verbose=FALSE, test.use="LR", only.pos=TRUE, latent.vars=c("sample"))' % (subsample, logfc_threshold))
        if saveMarkers:
            robjects.r('write.csv(cluster_markers_all, "cluster_markers_all_%s.csv")' % subsample)
    else:
        robjects.r('cluster_markers_all <- Seurat::FindAllMarkers(object=se, max.cells.per.ident=%s, assay=NULL, logfc.threshold=%s, min.pct=0.05, slot="counts", verbose=FALSE, test.use="wilcox", only.pos=TRUE)' % (subsample, logfc_threshold))
        if saveMarkers:
            robjects.r('write.csv(cluster_markers_all, "cluster_markers_all_%s_no.csv")' % subsample)
        
    return pd.DataFrame(robjects.r('cluster_markers_all')).set_index('gene')


def DEGonAnnData(ad_sel, identity='all_clusters_0.65', minPct=0.05, minCells=10, fdr=10**-2, nameAppend='', downsample=500, **kwargs):
    
    if not downsample is None:
        ad_sel = ad_sel[pd.concat([pd.Series(ad_sel.obs.index[ad_sel.obs[identity]==v]).sample(min(downsample, len(ad_sel.obs.index[ad_sel.obs[identity]==v])), random_state=0) for v in list(ad_sel.obs[identity].unique())]).values, :]
    
    ad_sel = ad_sel[:, ad_sel.var['is_human_gene'].values]
    
    vc = ad_sel.obs[identity].value_counts()
    ad_sel = ad_sel[ad_sel.obs[identity].isin(vc[vc >= minCells].index)]
    
    # Keep genes that are non-zero in at least <minPct>% of spots
    ad_sel = ad_sel[:, np.array((ad_sel.X>0).mean(axis=0))[0]>=minPct]
    print(ad_sel.shape, ad_sel.obs[identity].value_counts().to_dict())
    
    all_DEGs = DEG(ad_sel, identity=identity, **kwargs)
    print(all_DEGs.shape)

    temp = all_DEGs.loc[all_DEGs['p_val_adj']<=fdr].reset_index().set_index('cluster')['gene'].groupby(level=0).unique()
    print(temp.apply(len).to_dict(), temp.apply(len).sum())
    df_for_metascape = temp.apply(pd.Series).T
    df_for_metascape.columns = ['%s' % col for col in df_for_metascape.columns]
    df_for_metascape.to_csv('df_for_metascape_%s.csv' % nameAppend, index=False)
    
    return