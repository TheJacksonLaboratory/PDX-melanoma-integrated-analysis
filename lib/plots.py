
import os
import pickle
import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection, LineCollection

import scipy.stats

from numba import jit

@jit
def sMED(v1, v2):

    """Simple Minimum Event Distance (MED), similar (but not equivalent!) to the one implemented in MEDALT.
    MED is the minimal number and the series of single-copy gains or losses that are required to evolve one genome to another.
    """
    a = v1 - v2
    a = a[np.roll(a, -1) != a]
    a = np.abs(a[a != 0]).sum()
    return int(a)

def plotPhiHeatmap(gridPhi, X_grid_full, emb='emb', figsize=(5, 4), percentile=95.0, interpolation='bicubic',
                   cmapcolors=['black', 'maroon', 'red', 'white', 'orange', 'yellow', 'lightcyan'], aspect=None):

    fig, ax = plt.subplots(figsize=figsize)

    m = np.percentile(np.abs(gridPhi), percentile)

    cmap = colors.LinearSegmentedColormap.from_list('None', [colors.to_rgb(c) for c in cmapcolors], N=300)

    if aspect is None:
        aspect = (X_grid_full[:, :, 1].max() - X_grid_full[:, :, 1].min()) / (X_grid_full[:, :, 0].max() - X_grid_full[:, :, 0].min())

    im = ax.imshow(gridPhi.T[::-1,:], cmap=cmap, vmax=m, vmin=-m, interpolation=interpolation, aspect=aspect);
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlabel(f'{emb}_0')
    ax.set_ylabel(f'{emb}_1')

    plt.colorbar(mappable=im, ax=ax)

    return fig

def simpleCorrPlot(ida, idb, ax, ad_all, human_ratio=0.5, alpha=0.5):

    try:
        vah = ad_all[(ad_all.obs['sample']==ida) & (ad_all.obs['human_ratio']>=human_ratio), ad_all.var['is_human_gene']].to_df().T.apply(np.mean, axis=1).values
        vbh = ad_all[(ad_all.obs['sample']==idb) & (ad_all.obs['human_ratio']>=human_ratio), ad_all.var['is_human_gene']].to_df().T.apply(np.mean, axis=1).values
        ax.scatter(vah, vbh, s=2, alpha=alpha)
        mh = max(max(vah), max(vbh))
    except:
        mh = 0

    try:
        vam = ad_all[(ad_all.obs['sample']==ida) & (ad_all.obs['human_ratio']<human_ratio), ~ad_all.var['is_human_gene']].to_df().T.apply(np.mean, axis=1).values
        vbm = ad_all[(ad_all.obs['sample']==idb) & (ad_all.obs['human_ratio']<human_ratio), ~ad_all.var['is_human_gene']].to_df().T.apply(np.mean, axis=1).values
        ax.scatter(vam, vbm, s=2, alpha=alpha)
        mm = max(max(vam), max(vbm))
    except:
        mm = 0

    m = max(mh, mm)

    ax.plot([0, m], [0, m], '--', c='k')
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)

    ax.set_title(ida + '\n' + idb, fontsize=8)

    try:
        s, p = scipy.stats.pearsonr(vah, vbh)
        ax.text(0.05*m, 0.875*m, 'Pearson r: %s (h)' % (np.round(s, 3)), va='top', ha='left')
        #ax.text(0.05*m, 0.80*m, 'Pearson r: %s\np-value: %.2E (h)' % (np.round(s, 3), p), va='top', ha='left')
    except:
        pass

    try:
        s, p = scipy.stats.pearsonr(vam, vbm)
        ax.text(0.05*m, 0.95*m, 'Pearson r: %s (m)' % (np.round(s, 3)), va='top', ha='left')
        #ax.text(0.05*m, 0.95*m, 'Pearson r: %s\np-value: %.2E (m)' % (np.round(s, 3), p), va='top', ha='left')
    except:
        pass
    return

def simpleCorrPlotAll(ida, idb, ax, ad_all):

    va = ad_all[ad_all.obs['sample']==ida, :].to_df().T.apply(np.mean, axis=1).values
    vb = ad_all[ad_all.obs['sample']==idb, :].to_df().T.apply(np.mean, axis=1).values
    m = max(max(va), max(vb))
    ax.plot([0, m], [0, m], '--', c='k')
    ax.scatter(va, vb, s=2)
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_title(ida + '\n' + idb, fontsize=8)

    s, p = scipy.stats.pearsonr(va, vb)
    ax.text(0.05*m, 0.95*m, 'Pearson r: %s\np-value: %.2E' % (np.round(s, 3), p), va='top', ha='left')
    return

def plotSpatialAll(ads, ids = None, f = 0.75, nx = None, ny = None, panelSize = 5, fy=1.0, identity = None, palette = 'tab20', cmap = 'rainbow', title = None,
                   spot_size = 425, fontsize = 8, markerscale = 1.25, wspace = 0.15, hspace = 0.2, img_key = 'lowres',  x = 0.12, y = 0.93,
                   maintainScale = False, bsize = 100):

    if ids is None:
        ids = sorted(ads.keys())
    else:
        ads = {id:ads[id] for id in ids}

    n = len(ids)

    if not ny is None:
        print('Ignoring deprecated parameter ny=%s' % ny)

    if nx is None:
        nx = int(np.ceil(np.sqrt(n)))
            
    ny = int(np.ceil(n / nx))

    fig, axs = plt.subplots(ny, nx, figsize=(f*panelSize*nx, fy*f*panelSize*ny))
    if nx==1 and ny==1:
        axs = np.array([axs])

    if nx==1:
        axs = axs.reshape(ny, 1)
    elif ny==1:
        axs = axs.reshape(1, nx)

    if not identity is None:
        if identity in ads[ids[0]].obs.columns:
            is_cat = type(ads[ids[0]].obs[identity].dtype)==pd.core.dtypes.dtypes.CategoricalDtype
        else:
            is_cat = type(ads[ids[0]][:, identity].to_df()[identity].dtype)==pd.core.dtypes.dtypes.CategoricalDtype

    try:
        if identity in ads[ids[0]].obs.columns:
            ar = np.array([[(se := ads[id].obs[identity]).quantile(0.05), se.quantile(0.95)] for id in ads.keys()]).T
        else:
            ar = np.array([[(se := ads[id][:, identity].to_df()[identity]).quantile(0.05), se.quantile(0.95)] for id in ads.keys()]).T

        vmin, vmax = ar[0].min(), ar[1].max()
    except:
        vmin, vmax = 0, 1

    if maintainScale:
        maxSizex = 0
        maxSizey = 0
        for isample, sample in enumerate(ids):
            sizex, sizey, sizez = ads[sample].uns['spatial']['library_id']['images'][img_key].shape
            maxSizex = max(sizex, maxSizex) 
            maxSizey = max(sizey, maxSizey)

    for isample, sample in enumerate(ids):
        i, j = int((isample - isample%nx)/nx), isample%nx
        ax = axs[i, j]
        sf = ads[sample].uns['spatial']['library_id']['scalefactors']['tissue_%s_scalef' % img_key]

        if spot_size is None:
            spot_size = ads[sample].uns['spatial']['library_id']['scalefactors']['spot_diameter_fullres']

        sizex, sizey, sizez = ads[sample].uns['spatial']['library_id']['images'][img_key].shape
        
        if maintainScale:
            sizex, sizey = maxSizex, maxSizey
        else:
            sizex, sizey, sizez = ads[sample].uns['spatial']['library_id']['images'][img_key].shape

        sc.pl.spatial(ads[sample], img_key=img_key, color=identity, palette=palette, title=sample.replace('_', ' '), 
                        spot_size=spot_size, show=False, ax=ax, crop_coord=[0, sizey/sf, 0, sizex/sf], legend_loc='on data 2',
                        vmin=vmin, vmax=vmax, cmap=cmap)

        if not maintainScale:
            px = int(10**3*np.round((0.001*bsize/sf), 1))
            ax.text(0.05*sizey, 0.93*sizex-bsize/10, '%s px' % px, fontsize=8)
            ax.add_collection(PatchCollection([mpatches.Rectangle([0.05*sizey, 0.95*sizex-bsize/10], bsize, bsize/10, ec="none")], color='k', alpha=0.75))

        if not identity is None:
            if is_cat:
                # Otherwise it is a gene, for example, and not a categorical variable
                if identity in ads[sample].obs.columns:
                    for ilabel, label in enumerate(ads[sample].obs[identity].cat.categories):
                        color = ads[sample].uns[identity + '_colors'][ilabel]
                        ax.scatter([], [], color=color, label=label)

                    ax.legend(frameon=False, loc='upper left', fontsize=fontsize, markerscale=markerscale)

    for iext in range(isample+1, nx*ny):
        i, j = int((iext - iext%nx)/nx), iext%nx
        axs[i, j].axis('off')

    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    fig.suptitle(identity if title is None else title, x=x, y=y, ha='left', size='x-large', weight='demi')

    return


def plotSpatialEdge(ads, ids = None, f = 0.75, nx = None, ny = None, panelSize = 5, fy=1.0, identity = None, palette = 'tab20', cmap = 'rainbow', title = None,
                   spot_size = 425, fontsize = 8, markerscale = 1.25, wspace = 0.15, hspace = 0.2, img_key = 'lowres',  x = 0.12, y = 0.93,
                   maintainScale = False, bsize = 100, invert = False, decay_power = 3, absolute_cutoff=None, alpha_quantile = 0.5, method = 'MED',
                   linewidth_factor = 1.5, alpha=1.0, alpha_img=1.0, cacheEdges=False, cacheDir='cache', **kwargs):

    if ids is None:
        ids = sorted(ads.keys())
    else:
        ads = {id:ads[id] for id in ids}

    n = len(ids)

    if not ny is None:
        print('Ignoring deprecated parameter ny=%s' % ny)

    if nx is None:
        nx = int(np.ceil(np.sqrt(n)))
            
    ny = int(np.ceil(n / nx))

    fig, axs = plt.subplots(ny, nx, figsize=(f*panelSize*nx, fy*f*panelSize*ny))
    if nx==1 and ny==1:
        axs = np.array([axs])

    if nx==1:
        axs = axs.reshape(ny, 1)
    elif ny==1:
        axs = axs.reshape(1, nx)

    if not identity is None:
        if identity in ads[ids[0]].obs.columns:
            is_cat = type(ads[ids[0]].obs[identity].dtype)==pd.core.dtypes.dtypes.CategoricalDtype
        else:
            is_cat = type(ads[ids[0]][:, identity].to_df()[identity].dtype)==pd.core.dtypes.dtypes.CategoricalDtype

    try:
        if identity in ads[ids[0]].obs.columns:
            ar = np.array([[(se := ads[id].obs[identity]).quantile(0.05), se.quantile(0.95)] for id in ads.keys()]).T
        else:
            ar = np.array([[(se := ads[id][:, identity].to_df()[identity]).quantile(0.05), se.quantile(0.95)] for id in ads.keys()]).T

        vmin, vmax = ar[0].min(), ar[1].max()
    except:
        vmin, vmax = 0, 1

    if maintainScale:
        maxSizex = 0
        maxSizey = 0
        for isample, sample in enumerate(ids):
            sizex, sizey, sizez = ads[sample].uns['spatial']['library_id']['images'][img_key].shape
            maxSizex = max(sizex, maxSizex) 
            maxSizey = max(sizey, maxSizey)

    for isample, sample in enumerate(ids):
        i, j = int((isample - isample%nx)/nx), isample%nx
        ax = axs[i, j]
        sf = ads[sample].uns['spatial']['library_id']['scalefactors']['tissue_%s_scalef' % img_key]

        if spot_size is None:
            spot_size = ads[sample].uns['spatial']['library_id']['scalefactors']['spot_diameter_fullres']

        sizex, sizey, sizez = ads[sample].uns['spatial']['library_id']['images'][img_key].shape
        
        if maintainScale:
            sizex, sizey = maxSizex, maxSizey
        else:
            sizex, sizey, sizez = ads[sample].uns['spatial']['library_id']['images'][img_key].shape

        sc.pl.spatial(ads[sample], img_key=img_key, color=identity, palette=palette, title=sample.replace('_', ' '), 
                        spot_size=spot_size, show=False, ax=ax, crop_coord=[0, sizey/sf, 0, sizex/sf], legend_loc='on data 2',
                        vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, alpha_img=alpha_img)

        coords = ads[sample].obsm['spatial'].copy() * sf
        sspot_size = spot_size * sf
        
        cacheName = f'{cacheDir}/{sample}_{method}_edges_cache.pklz'

        if cacheEdges and os.path.isfile(cacheName):
            # Load from cache
            with open(cacheName, 'rb') as tempfile:
                egdes, edgeweights = pickle.load(tempfile)
        else:
            egdes = []
            edgeweights = []
            identities = []
            identitiesneigh = []
            for i, p in enumerate(coords):
                dist = np.sqrt(((coords - p)**2).sum(axis=1))
                cond = (dist > 0.5*sspot_size) & (dist <= 1.5*sspot_size)
                locs = np.where(cond)[0].tolist()
                cnv = ads[sample][ads[sample].obs.index[[i] + locs]].to_df().T

                clusters = ads[sample].obs['cluster'][locs].values.tolist()
                identities.extend(clusters)
                identitiesneigh.extend(ads[sample].obs['cluster'][[i]].values.tolist() * len(clusters))
                
                if method == 'MED':
                    weights = cnv.corr(method=sMED).iloc[0].iloc[1:].values
                else:
                    weights = 1 - cnv.corr(method=method).iloc[0].iloc[1:].values

                egdes.extend([(list(p), list(ep)) for ep in coords[locs]])
                edgeweights.extend(weights)

            if cacheEdges:
                if not os.path.exists(cacheDir):
                    os.makedirs(cacheDir)

                # Save to cache
                with open(cacheName, 'wb') as tempfile:
                    pickle.dump((egdes, edgeweights), tempfile, protocol=4)

        df_info = pd.concat([pd.Series(identities), pd.Series(identitiesneigh), pd.Series(edgeweights)], axis=1)

        alphas = pd.Series(edgeweights)
        alphas -= alphas.min()
        if absolute_cutoff is None:
            q = alphas.quantile(alpha_quantile)
        else:
            q = absolute_cutoff

        alphas /= q
        alphas[alphas > 1] = 1
        alphas = alphas.values**decay_power

        if invert:
            alphas = 1 - alphas

        ax.add_collection(LineCollection(egdes, linewidths=linewidth_factor*alphas, colors='k', linestyle='solid', alpha=alphas))

        if not maintainScale:
            px = int(10**3*np.round((0.001*bsize/sf), 1))
            ax.text(0.05*sizey, 0.93*sizex-bsize/10, '%s px' % px, fontsize=8)
            ax.add_collection(PatchCollection([mpatches.Rectangle([0.05*sizey, 0.95*sizex-bsize/10], bsize, bsize/10, ec="none")], color='k', alpha=0.75))

        if not identity is None:
            if is_cat:
                # Otherwise it is a gene, for example, and not a categorical variable
                if identity in ads[sample].obs.columns:
                    for ilabel, label in enumerate(ads[sample].obs[identity].cat.categories):
                        color = ads[sample].uns[identity + '_colors'][ilabel]
                        ax.scatter([], [], color=color, label=label)

                    ax.legend(frameon=False, loc='upper left', fontsize=fontsize, markerscale=markerscale)

    for iext in range(isample+1, nx*ny):
        i, j = int((iext - iext%nx)/nx), iext%nx

    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    fig.suptitle(identity if title is None else title, x=x, y=y, ha='left', size='x-large', weight='demi')
    plt.show()

    return df_info