
import os
import json
import numpy as np
import pandas as pd

import scanpy as sc
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib.colors

from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid
from skimage.draw import rectangle_perimeter, circle_perimeter, rectangle, disk
from skimage.util import unique_rows

def loadADs(ids, preprocessedDataPath):

    ads = {id: sc.read(preprocessedDataPath + '%s/st_adata_processed.h5ad' % id) for id in ids}

    for id in ids:
        ads[id].var['is_human_gene'] = ads[id].var['gene_ids'].str.startswith('ENSG').values
        ads[id].obs['original_barcode'] = ads[id].obs.index.values
        ads[id].obs['T'] = id.split('_')[1]
        ads[id].obs['treatment'] = 'Untreated'
        ads[id].obs.loc[ads[id].obs['T'].isin(['T1', 'T2', 'T3', 'T4']), 'treatment'] = 'Treated'
        ads[id].obs.index = ads[id].obs.index + '-' + id
        ads[id].obs['sample'] = id
    
        ads[id].obs['human_count'] = np.exp(ads[id][:, ads[id].var['is_human_gene']].to_df()).sub(1.).sum(axis=1)
        ads[id].obs['mouse_count'] = np.exp(ads[id][:, ~ads[id].var['is_human_gene']].to_df()).sub(1.).sum(axis=1) 
        ads[id].obs['human_ratio'] = ads[id].obs['human_count'] / (ads[id].obs['mouse_count'] + ads[id].obs['human_count'])
        ads[id].obs['mouse_ratio'] = ads[id].obs['mouse_count'] / (ads[id].obs['mouse_count'] + ads[id].obs['human_count'])

    return ads

def scaleMH(ads, ids, total=2*10**4, L=5*10**4, figsize=(7,5), human_ratio=0.8, quantile=0.5, palette=None):
    
    ad_sel = sc.concat({id:ads[id] for id in ids}, label='sample', join='outer', index_unique ='-')
    v_unscaled = (np.exp(ad_sel.to_df()).sub(1.)).sum(axis=1).values

    print('Sample', '\t', 'Median (across spots) total mRNA, before scaling, human spots')
    for id in ids:
        ma = np.exp(ads[id][ads[id].obs['human_ratio']>=human_ratio, ads[id].var['is_human_gene'].values].to_df()).sub(1.).sum(axis=1).quantile(quantile)
        ma = total if ma is np.nan else ma
        ads[id].X = csr_matrix(np.log(((np.exp(ads[id].to_df()).sub(1.))*total/ma).add(1.)).values)
        print(id, '\t', np.round(ma, 0))
        
    for id in ids:
        ads[id].obs['human_count_scaled'] = np.exp(ads[id][:, ads[id].var['is_human_gene']].to_df()).sub(1.).sum(axis=1)
        ads[id].obs['mouse_count_scaled'] = np.exp(ads[id][:, ~ads[id].var['is_human_gene']].to_df()).sub(1.).sum(axis=1)

    ad_sel = sc.concat({id:ads[id] for id in ids[:]}, label='sample', join='outer', index_unique ='-')
    v_scaled = (np.exp(ad_sel.to_df()).sub(1.)).sum(axis=1).values
    
    ad_sel.obs['total_counts_non_scaled'] = v_unscaled
    ad_sel.obs['total_counts_scaled'] = v_scaled

    fig, ax = plt.subplots(figsize=figsize)
    g = 'treatment'
    sc.pl.violin(ad_sel, keys='total_counts_non_scaled', groupby=g, palette={v: 'azure' for i, v in enumerate(ad_sel.obs[g].unique())}, stripplot=False, show=False, ax=ax, alpha=0.5)
    sc.pl.violin(ad_sel, keys='total_counts_scaled', groupby=g, palette=palette, stripplot=False, show=False, ax=ax)
    ax.set_ylim([0, L])

    fig, ax = plt.subplots(figsize=figsize)
    g = 'T'
    sc.pl.violin(ad_sel, keys='total_counts_non_scaled', groupby=g, palette={v: 'azure' for i, v in enumerate(ad_sel.obs[g].unique())}, stripplot=False, show=False, ax=ax, alpha=0.5)
    sc.pl.violin(ad_sel, keys='total_counts_scaled', groupby=g, palette=palette, stripplot=False, show=False, ax=ax)
    ax.set_ylim([0, L])

    fig, ax = plt.subplots(figsize=figsize)
    g = 'sample'
    sc.pl.violin(ad_sel, keys='total_counts_non_scaled', groupby=g, palette={v: 'azure' for i, v in enumerate(ad_sel.obs[g].unique())}, stripplot=False, show=False, ax=ax, alpha=0.5, rotation=90)
    sc.pl.violin(ad_sel, keys='total_counts_scaled', groupby=g, palette=palette, stripplot=False, show=False, ax=ax, rotation=90)
    ax.set_ylim([0, L])
    
    ad_sel = sc.concat({id:ads[id][ads[id].obs['human_ratio']>=human_ratio, ads[id].var['is_human_gene'].values] for id in ids}, label='sample', join='outer', index_unique ='-')
    
    v_scaled = (np.exp(ad_sel.to_df()).sub(1.)).sum(axis=1).values
    
    ad_sel.obs['total_counts_scaled'] = v_scaled   
    
    fig, ax = plt.subplots(figsize=figsize)
    g = 'sample'
    sc.pl.violin(ad_sel, keys='total_counts_scaled', groupby=g, palette=palette, stripplot=False, show=False, ax=ax, rotation=90)
    ax.set_ylim([0, L])    
    
    return {id:v for id, v in zip(ids, v_unscaled)}

def concatADs(ads, ids):

    ad_all = sc.concat({id:ads[id] for id in ids}, join='outer')
    df_var = pd.concat([ads[id].var for id in ids])
    df_var = df_var.loc[~df_var.index.duplicated()]
    ad_all.var = df_var.reindex(ad_all.var.index)

    print(ad_all.shape)

    return ad_all

def swapLabels(ad, a, b, id='all_clusters_0.00'):
    
    wha = ad.obs[id]==a
    whb = ad.obs[id]==b
    
    ad.obs[id][wha] = b
    ad.obs[id][whb] = a
    
    return

def loadAdImage(samplePath):

    thumbnail = plt.imread(samplePath + '/thumbnail.tiff')

    with open(samplePath + '/grid/grid.json', 'r') as f:        
        d = json.load(f)

    grid = pd.read_csv(samplePath + '/grid/grid.csv', index_col=0, header=None)

    image = {'library_id': {'images': {'lowres': thumbnail},
                                'metadata': {'chemistry_description': None, 'software_version': None},
                                'scalefactors': {'tissue_lowres_scalef': thumbnail.shape[0]/d['y'],
                                                 'spot_diameter_fullres': d['spot_diameter_fullres']}}}, grid.index.values, grid[[5, 4]].values
    return image

def loadImFeatures(preprocessedImAdDataPath):

    df_temp = pd.read_csv(preprocessedImAdDataPath + '/data.csv.gz', index_col=[0,1]).xs(1, level='in_tissue')
    df_temp.insert(0, 'original_barcode', df_temp.index.values)
    ad = sc.AnnData(X=df_temp.loc[:, df_temp.columns.str.contains('feat')],
                    obs=df_temp.loc[:, ~df_temp.columns.str.contains('feat')])

    return ad


def getDFFromAD(ad, identity, ensembl=False):
    
    df_temp = ad.to_df().T
    df_temp.columns = pd.MultiIndex.from_frame(ad.obs.reset_index()[['index', 'sample', identity]], names=['spot', 'sample', 'cluster'])

    if ensembl:
        df_temp.index = pd.Index(ad.var['gene_ids'], name='ensembl')
    else:
        df_temp.index = pd.Index(ad.var.reset_index()['index'], name='symbol')
    
    df_temp = df_temp.loc[(df_temp>0).mean(axis=1)>0]
    
    return df_temp

def prepareExpressionForInferCNV(ad_all, model, dataPath, identity='all_clusters_0.65'):

    df= getDFFromAD(ad_all[:, ad_all.var['is_human_gene']], identity)
    print('%s genes, %s spots' % df.shape)
    
    # Export spots metadata
    columns = df.columns
    df_columns = columns.to_frame().droplevel(['sample', 'cluster'])[['sample', 'cluster']]
    df_columns['group.name'] = df_columns['sample'].astype('str') + '.cluster' + df_columns['cluster'].astype('str')
    df_columns = df_columns.drop(['sample', 'cluster'], axis=1)
    df_columns.index.name = 'sample.name'
    df_columns = df_columns.reset_index()
    sel = df_columns['group.name'].value_counts()[df_columns['group.name'].value_counts()>1].index
    df_columns = df_columns[df_columns['group.name'].isin(sel.values)]
    df_columns.to_csv(dataPath + 'For_inferCNV_%s_meta.data.tsv.gz' % model, sep='\t', index=False, header=False)    

    # Export gene expression
    df = df.droplevel(['cluster', 'sample'], axis=1)
    df = df.loc[:, df.columns.get_level_values('spot').isin(df_columns['sample.name'])]

    pd.Series(df.index).to_csv(dataPath + 'For_inferCNV_%s_genes.tsv.gz' % model, sep='\t', index=False, header=False)

    df.to_csv(dataPath + 'For_inferCNV_%s_gene.expression.tsv.gz' % model, sep='\t')
  
    return


def prepareExpressionForCasperCNV(ad_all, model, dataPath, identity='all_clusters_0.65'):

    df= getDFFromAD(ad_all[:, ad_all.var['is_human_gene']], identity, ensembl=True)
    print('%s genes, %s spots' % df.shape)

    # Export spots metadata 
    df_columns = df.columns.to_frame().droplevel(['sample', 'cluster'])[['sample', 'cluster']]
    df_columns['loh.name'] = df_columns['sample'].astype('str') + '_human'
    df_columns['group.name'] = df_columns['sample'].astype('str') + '.cluster' + df_columns['cluster'].astype('str')
    df_columns = df_columns.drop(['sample', 'cluster'], axis=1)
    df_columns.index.name = 'sample.name'
    df_columns.to_csv(dataPath + 'For_CasperCNV_%s_loh.name_group.name.csv.gz' % model)

    # Export gene expression
    df = df.droplevel(['cluster', 'sample'], axis=1)
    df.to_csv(dataPath + 'For_CasperCNV_%s_gene.expression.csv.gz' % model)
    
    return

def loadCNVfromCaSpER(lohfile, cnvfile):
    df_meta = pd.read_csv(lohfile, index_col=0)
    df_meta.columns = ['sample', 'cluster']
    df_meta['sample'] = df_meta['sample'].apply(lambda s: s[:-6])
    df_meta['cluster'] = df_meta['cluster'].str.split('cluster', expand=True)[1].astype(str)
    df_meta['time'] = df_meta['sample'].str.split('_', expand=True)[1].astype(str)
    df_meta['sample'] = df_meta['sample'].apply(lambda s: s[7:12])
    
    df = pd.read_csv(cnvfile, index_col=0)
    
    df.columns = pd.MultiIndex.from_frame(df_meta.loc[df.columns].reset_index().rename({'index': 'spot'}, axis=1))
    df = df.loc[~df.index.isna()]
    
    df_obs = df.columns.to_frame()[['sample', 'cluster', 'time']].droplevel(['sample', 'cluster', 'time']).astype(str).astype('category')
    print(df.shape, df_obs.shape)
    return df, df_obs

def loadCNVfromInferCNV(metafile, files, discretize=True, delta=0.025):
    dfm = pd.read_csv(metafile, sep='\t', header=None)
    index = dfm[0].values
    dfm = dfm[1].str.split('.', expand=True)
    dfm['cluster'] = dfm[1].str.split('cluster', expand=True)[1]
    dfm['sample'] = dfm[0].apply(lambda s: s[7:12])
    dfm['time'] = dfm[0].str.split('_', expand=True)[1]
    dfm = dfm[['sample', 'cluster', 'time']]
    dfm.index = index
    dfm.index.name = 'spot'
    
    df = pd.concat([pd.read_csv(file, sep=' ', index_col=0) for file in files], axis=1)
    print(df.shape, dfm.shape)
    
    dfm = dfm.loc[df.columns]
    
    dfm.index.name = 'spot'
    df.columns.name = 'spot'
    
    if discretize:
        wh_neg = df < 1-delta
        df[df < 1+delta] = 0
        df[wh_neg] = -1
        df[df >= 1+delta] = 1
        df = df.astype(int)  

    return df, dfm

def exportClustersForQuPath(sample, samplePathNF2, thumbsDir, obs, palette, useROIfile=False):
    dims_json_name = [name for name in os.listdir(thumbsDir) if ('.json' in name) and (sample in name.replace(' ', '_')) and ('dimensions' in name)][0]
    #dims_json_name = [name for name in os.listdir(thumbsDir) if ('.json' in name) and (sample[:-2] in name) and ('dimensions' in name)][0]
    with open(thumbsDir + dims_json_name, 'r') as outfile:
        infodim = json.loads(outfile.read())
    x, y = infodim['x'], infodim['y']
    print('x, y:', x, y)

    grid_json_name = samplePathNF2 + 'grid/grid.json'
    with open(grid_json_name, 'r') as outfile:
        infogrid = json.loads(outfile.read())
    xc, yc = infogrid['x'], infogrid['y']
    print('xc, yc:', xc, yc)

    if useROIfile:
        roi_json_name = [name for name in os.listdir(thumbsDir) if ('.json' in name) and (sample in name.replace(' ', '_')) and ('.oid' in name)][0]
        #roi_json_name = [name for name in os.listdir(thumbsDir) if ('.json' in name) and (sample[:-2] in name) and (('.oid%s.' % sample[-1]) in name)][0]
        with open(thumbsDir + roi_json_name, 'r') as outfile:
            inforoi = json.loads(outfile.read())

        scalex = inforoi['0']['size']*x/xc
        scaley = inforoi['1']['size']*y/yc
        shiftx = inforoi['0']['location']*x
        shifty = inforoi['1']['location']*y
    else:
        scalex = 1
        scaley = 1
        shiftx = 0
        shifty = 0

    features = []
    df_temp = obs[['cluster', 'pxl_col_in_fullres', 'pxl_row_in_fullres']].set_index('cluster')

    df_temp['pxl_col_in_fullres'] *= scalex
    df_temp['pxl_row_in_fullres'] *= scaley

    df_temp['pxl_col_in_fullres'] += shiftx
    df_temp['pxl_row_in_fullres'] += shifty

    se = df_temp.apply(lambda s: [s[0], s[1]], axis=1).groupby(level=0).agg(lambda s: np.vstack(s.values).tolist())
    for cluster in se.index:
        color = (np.array(matplotlib.colors.to_rgb(palette[cluster]))*255).astype(int).tolist()
        subdict = dict()
        subdict.update({'type': 'Feature'})
        subdict.update({'geometry': {'type': 'MultiPoint', 'coordinates': se.loc[cluster]}})
        subdict.update({'properties': { 'object_type': 'annotation', 'name': 'Cluster %s' % cluster, 'color': color, 'isLocked': True}})
        features.append(subdict)

    gjson = {'type': 'FeatureCollection', 'features': features}

    with open(f'{sample}.clusters.geojson', 'w') as tempfile:
        tempfile.write(json.dumps(gjson))

    return

from skimage.draw import disk, rectangle

from skimage.draw import polygon

def makeMaskFromQuPathAnnotations(dims, jsonpath, filter=None):
    
    # {'type': str, 'features': [{'type': str, 'geometry': {'type': str, 'coordinates': list(1, xxx, 2)}, 'properties': {'object_type': str, 'isLocked': bool}}, ...]}
    with open(jsonpath, 'r') as outfile:
        data = json.loads(outfile.read())

    mask = np.zeros((dims[0], dims[1])).astype(np.uint8)

    polycount = 0
    for wblock in data['features']:
        obj = wblock['geometry']['coordinates']
        
        if isinstance(obj[0][0][0], list):
            # This is a multipolypon
            for p in obj:
                a = np.array(p[0])
                rr, cc = polygon(a[:, 1], a[:, 0])
                mask[rr, cc] = 1
                polycount += 1
        else:
            # This is a polygon
            a = np.array(obj[0])
            rr, cc = polygon(a[:, 1], a[:, 0])
            mask[rr, cc] = 1
            polycount += 1

    print('Loaded %s objects (%s polygons)' % (len(data['features']), polycount))
    
    return mask

def getMaskFromMetadata(df_metadata, tile_size=None, target_shape=None):

    mask = np.zeros((target_shape[0], target_shape[1])).astype(np.uint8)
    print(mask.shape)
    tile_half_size = int((tile_size - 1) / 2)
    coords = df_metadata[['col', 'row', 'mask']].values
    for i in range(len(coords)):
        if coords[i][2]==1:
            cc, rr = rectangle(start=(coords[i][0] - tile_half_size, coords[i][1] - tile_half_size),
                               end=  (coords[i][0] + tile_half_size, coords[i][1] + tile_half_size),
                               shape=(mask.shape[1], mask.shape[0]))
            mask[rr, cc] = 1

    return mask

def mapMaskToIiles(mask, df_metadata=None, tile_size=None, cutoff=0.5):

    tile_half_size = int((tile_size - 1) / 2)
    coords = df_metadata[['col', 'row']].values
    out = []
    for i in range(len(coords)):
        cc, rr = rectangle(start=(coords[i][0] - tile_half_size, coords[i][1] - tile_half_size),
                           end=  (coords[i][0] + tile_half_size, coords[i][1] + tile_half_size),
                           shape=(mask.shape[1], mask.shape[0]))
        fraction_filled = 1 if mask[rr, cc].ravel().mean() >= cutoff else 0
        out.append(fraction_filled)

    return pd.Series(index=df_metadata.index, data=out)

def getMaskFromAd(ad, identity=None, imageid='lowres', target_shape=None):

    sf = ad.uns['spatial']['library_id']['scalefactors'][f'tissue_{imageid}_scalef']
    tile_size = int(np.ceil(ad.uns['spatial']['library_id']['scalefactors']['spot_diameter_fullres'] * sf))
    tile_size += 1 if tile_size % 2 == 0 else 0
    if identity is None:
        df_temp = ad.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].copy()
        df_temp.columns = ['row', 'col']
        df_temp['mask'] = 1
    else:
        df_temp = ad.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres', identity]].copy()
        df_temp.columns = ['row', 'col', 'mask']        
    df_temp['row'] = (df_temp['row'] * sf).astype(int)
    df_temp['col'] = (df_temp['col'] * sf).astype(int)
    df_temp['mask'] = df_temp['mask'].replace({'0': 1, '-1': 0})
    if target_shape is None:
        target_shape = ad.uns['spatial']['library_id']['images'][imageid].shape

    return getMaskFromMetadata(df_metadata=df_temp, tile_size=tile_size, target_shape=target_shape)

def mapMaskToAd(mask, ad, identity='from_mask', imageid='lowres', cutoff=0.5):

    sf = ad.uns['spatial']['library_id']['scalefactors'][f'tissue_{imageid}_scalef']
    tile_size = int(np.ceil(ad.uns['spatial']['library_id']['scalefactors']['spot_diameter_fullres'] * sf))
    tile_size += 1 if tile_size % 2 == 0 else 0
    df_temp = ad.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].copy()
    df_temp.columns = ['row', 'col']
    df_temp['row'] = (df_temp['row'] * sf).astype(int)
    df_temp['col'] = (df_temp['col'] * sf).astype(int)
    se = mapMaskToIiles(mask, df_metadata=df_temp, tile_size=tile_size, cutoff=cutoff)
    ad.obs[identity] = se.values

    return 

def computFullGridVelocity(ad, emb):
    
    """emb: 'spatial', 'umap', ...
    
    Note: compute_velocity_on_grid returns stacked and filtered values
    We need to generate the full grid.
    """

    X_emb = np.array(ad.obsm[f'X_{emb}'][:, [0,1]])
    V_emb = np.array(ad.obsm[f'velocity_{emb}'][:, [0,1]])

    X_grid, V_grid = compute_velocity_on_grid(
                X_emb=X_emb,
                V_emb=V_emb,
                density=1,
                autoscale=True,
                smooth=None,
                n_neighbors=None,
                min_mass=None)

    ux = np.unique(X_grid.T[0])
    dx = ux[1] - ux[0]
    uy = np.unique(X_grid.T[1])
    dy = uy[1] - uy[0]
    delta = np.array([dx, dy])

    shift = X_grid.min(axis=0)
    loc = ((X_grid - shift) / delta[None, None, :]).round(0).astype(int)[0].T

    shape = *(loc.max(axis=1)+1), 2
    grid_full = np.zeros(shape)

    X = np.array([np.hstack([[i]*shape[1] for i in range(shape[0])]), list(range(shape[0]))*shape[1]]).T.reshape(shape[0], shape[1], 2).astype(float)
    X *= delta
    X += shift

    V = grid_full.copy()
    V[loc[0], loc[1]] = V_grid
    
    return X, V, X_emb

def getGridPhi(X, V, R=2, regionShape='disk'):
    
    """regionShape: 'square' or 'disk'
    For R=1 or 2 or 3 the coordinates are the same as rectangle
    """
    
    shape = X.shape
    Phi = np.zeros((shape[0], shape[1]))
    for i0 in range(shape[0]):
        for j0 in range(shape[1]):
            if False:
                ii, jj = rectangle_perimeter(start=(i0-R+1, j0-R+1), extent=(2*R-1, 2*R-1), shape=shape)
                ii, jj = unique_rows(np.vstack([ii, jj]).T).T
                #ii, jj = circle_perimeter(i0, j0, R, shape=shape) # code not tested
            else:
                if regionShape == 'square':
                    ii, jj = rectangle(start=(i0-R, j0-R), extent=(2*R+1, 2*R+1), shape=shape)
                    ii, jj = np.hstack(ii), np.hstack(jj)
                elif regionShape == 'disk':
                    ii, jj = disk((i0, j0), R+1, shape=shape)
                else:
                    raise NotImplementedError

            d = X[ii, jj]  - X[i0, j0]
            w = (np.sign(d)) * V[ii, jj]
            Phi[i0, j0] = w.sum()

    Phi /= max(np.abs(Phi.max()), np.abs(Phi.min()))

    return Phi

def getFiOfEmb(X, Phi, X_emb):
    
    X_1D = X.reshape(X.shape[0]*X.shape[1], 2)
    Phi_1D = Phi.reshape(Phi.shape[0]*Phi.shape[1], 1)
    
    match = np.array([np.argmin(((X_1D - obs)**2).sum(axis=1)) for obs in X_emb])
    Phi_emb = Phi_1D[match].T[0]
    
    return Phi_emb

def getGridVelocityForSourceSink(X_grid_full, V_grid_full, gridPhi, isSource=True):

    shape = V_grid_full.shape
    
    V_grid_full_temp = V_grid_full.copy()
    resc = gridPhi.copy()
    
    if isSource:
        resc[gridPhi<0] = 0
    else:
        resc[gridPhi>0] = 0
    
    V_grid_full_temp[:, :, 0] = 0
    V_grid_full_temp[:, :, 1] = resc

    X_grid_full_1D = X_grid_full.reshape(shape[0]*shape[1], 2)
    V_grid_full_1D = V_grid_full_temp.reshape(shape[0]*shape[1], 2)

    wh = ~(V_grid_full_1D==0).all(axis=1)
    X_grid_full_1D = X_grid_full_1D[wh, :]
    V_grid_full_1D = V_grid_full_1D[wh, :]
    
    return X_grid_full_1D, V_grid_full_1D

def remove_up_to(ins, n=50):
    
    s = ins.copy()
    
    t = s - np.roll(s.values, -1)
    a = np.where(t[:-1]==1)[0]
    b = a + 1
    t = s - np.roll(s.values, 1)
    a = np.where(t[:-1]==1)[0]
    a = a[:len(b)]
    b = b[:len(a)]
    b -= a
    
    for i, l in np.vstack([a, b]).T:
        if l<=n:
            s[i: i+l] = 0

    return s

def filterCNV(df, n=50):

    f = lambda s: remove_up_to(s, n)

    df_pos = df.copy()
    df_pos[df_pos < 0] = 0
    df_pos = df_pos.apply(f, axis=0)

    df_neg = df.copy()
    df_neg[df_neg > 0] = 0
    df_neg *= -1
    df_neg = df_neg.apply(f, axis=0)
    df_neg *= -1

    return df_pos + df_neg