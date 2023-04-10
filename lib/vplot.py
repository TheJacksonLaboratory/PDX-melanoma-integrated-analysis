
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from scanpy.tools._score_genes import score_genes

def vplot(df, dfc, name='Name', figsize=(20,5), palette=None, step=None, q=[0, 1], psteps=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
          nlines=10, addZeroLine=False, rightylabel=False, saveName=None, vmin=None, vmax=None):
    
    tvmin, tvmax = df.stack().replace([np.inf, -np.inf], np.nan).dropna().quantile(q).values

    if vmin is None:
        vmin = tvmin

    if vmax is None:
        vmax = tvmax
    
    if step is None:
        psteps = np.array(psteps)
        step = (vmax - vmin) / nlines
        step = psteps[np.argmin(np.abs((psteps / step) - 1))]
    
    vmax = step*np.ceil(np.array([vmax/step]))
    vmin = step*np.floor(np.array([vmin/step]))
    
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    def set_axis_style(ax, labels):
        ax.xaxis.set_tick_params(direction='out', rotation=0) # 90
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        #ax.set_xlabel('Sample name')
        if rightylabel:
            ax.set_ylabel(name.replace('_', ' '), rotation=-90, va='bottom')
            ax.yaxis.set_label_position('right')
            ax.yaxis.set_label_coords(1.02, 0.5)
        else:
            ax.set_ylabel(name.replace('_', ' '), va='bottom')
        #ax.tick_params(axis='y', which='both', rotation=0, labelleft=True, labelright=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    #ax.set_title(name)
    
    for p in np.linspace(vmin, vmax, int(np.round(((vmax-vmin)/step), 0))+1)[1:-1]:
        ax.axhline(p, linestyle='--', linewidth=1, c='grey', alpha=0.5, zorder=-np.inf)
        
    if addZeroLine:
        ax.axhline(0, linestyle='-', linewidth=1, c='k', alpha=0.75, zorder=-np.inf)
    
    vdata = [df.values.T[i][~np.isnan(df.values.T[i])] for i in range(df.shape[1])]
    cdata = [dfc.values.T[i][~np.isnan(dfc.values.T[i])] for i in range(dfc.shape[1])]
    
    meta_samples = []
    meta_clusters = []
    meta_len = []
    ndata = []
    for i, (svdata, scdata) in enumerate(zip(vdata, cdata)):
        tdata = pd.Series(index=scdata, data=svdata).replace([np.inf, -np.inf], np.nan).dropna().groupby(level=0).apply(list)
        ndata.extend(tdata.values)
        
        temp_clusters = tdata.index.astype(int).astype(str)
        meta_clusters.extend(temp_clusters)
        
        temp_samples = [df.columns[i]]*len(temp_clusters)
        meta_samples.extend(temp_samples)
        
        meta_len.append(len(temp_clusters))
        
    parts = ax.violinplot(ndata, showmeans=False, showmedians=False, showextrema=False)
    
    pos = np.cumsum(meta_len) + 0.5
    for p in pos[:-1]:
        ax.axvline(p, linestyle='-', linewidth=1, c='k', alpha=0.5)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(palette[meta_clusters[i]])
        pc.set_edgecolor('grey')
        pc.set_alpha(1)

    quartile1 = [np.percentile(v, 25) for v in ndata]
    medians = [np.percentile(v, 50) for v in ndata]
    quartile3 = [np.percentile(v, 75) for v in ndata]

    whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(ndata, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=4, edgecolor='k')
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=3)
    ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    labels = [cluster for sample, cluster in zip(meta_samples, meta_clusters)]
    #labels = ['' + sample.split('_')[2][:] for sample, cluster in zip(meta_samples, meta_clusters)]
    #labels = [sample.replace('_', ' ') + ': ' + cluster for sample, cluster in zip(meta_samples, meta_clusters)]
    set_axis_style(ax, labels)
    
    ax.set_ylim([vmin, vmax])
    
    lims = ax.get_ylim()
    tmin = lims[0] + (lims[1] - lims[0])*0.025
    tmax = lims[1] - (lims[1] - lims[0])*0.075
    
    #for i, c in enumerate(meta_clusters):
    #    ax.text(i+1, tmin, c, ha='center')
    
    pos = []
    last_pos = 0
    for l in meta_len:
        last_pos += l/2 
        pos.append(last_pos)
        last_pos += l/2 
        
    for i, s in enumerate(pos):
        label = df.columns[i].split('_')[1] + '\n' + df.columns[i].split('_')[2]
        if label == '1':
            label = 'C4'
        ttext = ax.text(pos[i]+0.5, tmax, label, ha='center')
        ttext.set_path_effects([path_effects.Stroke(linewidth=6, foreground='w'),path_effects.Normal()])

    fig.tight_layout()

    if not saveName is None:
        plt.savefig(saveName + '.png', dpi=300, facecolor='w')

    plt.show()
    
    return

def addLogRatioOfScores(score1, score2, shift=0.01, ads=None, ids=None, vdivider='_over_', vprefix='score_'):
    
    name = '%s%s%s' % (score1, vdivider, score2)
    
    for id in ids:
        ad = ads[id]
        
        s1 = ad.obs[vprefix + score1].copy()
        s1 -= s1.min()
        s1 = s1.replace(0, shift)
        
        s2 = ad.obs[vprefix + score2].copy()
        s2 -= s2.min() - shift
        s2 = s2.replace(0, shift)

        ad.obs[vprefix + name] = np.log(s1 / s2)
    
    return name

def wrapVplotRatio(score1, score2, genesDicts=None, ads=None, ids=None, identity=None, palette=None, vdivider='_over_', vprefix='score_', **kwargs):

    checkAddScore(score1, genes=genesDicts[score1], ads=ads, ids=ids, vprefix=vprefix)
    checkAddScore(score2, genes=genesDicts[score2], ads=ads, ids=ids, vprefix=vprefix)

    score = addLogRatioOfScores(score1, score2, ads=ads, ids=ids, vprefix=vprefix, vdivider=vdivider)

    return wrapVplotScore(score, genesDicts={score: []}, addZeroLine=True, rightylabel=True, identity=identity, palette=palette, ads=ads, ids=ids, vprefix=vprefix, **kwargs)

def wrapVplotScore(score, genesDicts=None, identity=None, selected=[], palette=None, ads=None, ids=None, vprefix='score_', **kwargs):
    
    checkAddScore(score, genes=genesDicts[score], ads=ads, ids=ids, vprefix=vprefix)

    if len(selected)==0:
        selected = pd.Series(np.hstack([ads[id].obs[identity] for id in ids])).sort_values().unique().tolist()

    df = pd.concat([pd.Series(ads[id][ads[id].obs[identity].isin(selected), :].obs[vprefix + score].fillna(0).values) for id in ids], keys=ids, axis=1)
    dfc = pd.concat([pd.Series(ads[id][ads[id].obs[identity].isin(selected), :].obs[identity].values.astype(int)) for id in ids], keys=ids, axis=1)
    
    return vplot(df, dfc, name=vprefix + score, palette=palette, **kwargs)

def wrapVplotGene(gene, identity=None, selected=[], palette=None, ads=None, ids=None, vprefix='gene_', **kwargs):

    if len(selected)==0:
        selected = pd.Series(np.hstack([ads[id].obs[identity] for id in ids])).sort_values().unique().tolist()
    
    allGenes = pd.concat([ads[id][ads[id].obs[identity].isin(selected), :].var for id in ids]).index.unique()
    if not gene in allGenes:
        print('Gene not found:', gene)
        return allGenes

    df = pd.concat([pd.Series(ads[id][ads[id].obs[identity].isin(selected), gene].to_df()[gene].values) for id in ids], keys=ids, axis=1)
    dfc = pd.concat([pd.Series(ads[id][ads[id].obs[identity].isin(selected), :].obs[identity].values.astype(int)) for id in ids], keys=ids, axis=1)
    
    return vplot(df, dfc, name=vprefix + gene, palette=palette, **kwargs)

def checkAddScore(score, genes=None, ads=None, ids=None, vprefix='score_', ctrl=50, verbose=0):

    for id in ids:
        if not vprefix + score in ads[id].obs.columns:
            if verbose>0:
                print('Scoring genes:', id, score, len(genes), len(ads[id].var.index.intersection(genes)))
            score_genes(ads[id], 
                        gene_list=ads[id].var.index.intersection(genes), 
                        score_name=vprefix + score,
                        ctrl_size=ctrl)

    return