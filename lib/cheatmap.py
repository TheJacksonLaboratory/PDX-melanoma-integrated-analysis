
import os
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import scipy.cluster.hierarchy
from scipy.spatial.distance import cdist, pdist, squareform

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

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

def cplot(df, df_groups, groupby=None, figsize=(20,15), clusterVar=True, clusterObs=True, p=4, palette=None, colormap=plt.cm.bwr, colormapBad='grey', minCellsPerGroup=15,
          addGeneLabels=False, fontsizeGeneLabels=12, saveFig=False, borderWidth=0.005, groupWidth=0.0375, dendrogramLineWidth=2.0, dendrogramLineColor='#555555', safetyLimit=1000,
          addLinesOnHeatmap=True, addLinesOnGroups=True, vmin=None, vmax=None, returnDistancesOnly=False, bootstrapObs=False, bootstrapNum=100,
          linkageMethod='ward', keepOriginalOrderObs=False, linkageMetric='euclidean', # 'correlation', 'cosine', 'euclidean'
          useMEDforObs=False, optimalOrderingForObs=False, useMEDforObsGroups=False, sampleMED=50, groupfontsize=10,
          clusterObsByGroups=True, reference=dict(), referenceLabel='Reference', colorbarLabels=None, colorbarLabel='Gene Expression', figureName='figure.png', dpi=600):
    
    # colorbarLabels: ['High', 'Low'], ['Ampl.', 'Del.'], None

    #print('Before filtering:', df.shape)
    cols = df_groups.columns.values.tolist()
    temp_sel = df_groups.reset_index().set_index(cols).groupby(level=list(range(len(cols))))['spot'].count()
    df_groups = df_groups.loc[pd.Index(df_groups.values).isin(temp_sel[temp_sel>=minCellsPerGroup].index)]
    df = df[df_groups.index]
    print('After filtering:', df.shape)
    
    # translate_pos was copied from scanpy:
    # https://github.com/scverse/scanpy/blob/2e98705347ea484c36caa9ba10de1987b09081bf/scanpy/plotting/_anndata.py#L2247
    def translate_pos(pos_list, new_ticks, old_ticks):
            if not isinstance(old_ticks, list):
                # assume that the list is a numpy array
                old_ticks = old_ticks.tolist()
            new_xs = []
            for x_val in pos_list:
                if x_val in old_ticks:
                    new_x_val = new_ticks[old_ticks.index(x_val)]
                else:
                    # find smaller and bigger indices
                    idx_next = np.searchsorted(old_ticks, x_val, side="left")
                    idx_prev = idx_next - 1
                    old_min = old_ticks[idx_prev]
                    old_max = old_ticks[idx_next]
                    new_min = new_ticks[idx_prev]
                    new_max = new_ticks[idx_next]
                    new_x_val = ((x_val - old_min) / (old_max - old_min)) * (
                        new_max - new_min
                    ) + new_min
                new_xs.append(new_x_val)
            return new_xs
        
    for col in df_groups.columns:
        try:
            df_groups[col].cat.remove_unused_categories(inplace=True)
        except:
            pass
           
    assert (df.columns==df_groups.index).all()
    
    if not returnDistancesOnly:
        fig = plt.figure(figsize=figsize)
    
    b = borderWidth
    xa, xb, xc, xd, xe = 0.05, 0.15, 0.75, 0.85, 0.95
    ya, yc, yb, yd, ye = 0.05, 0.25, 0.65, 0.93, 0.95
    
    groupw = groupWidth
    if clusterVar:
        xb = groupw * df_groups.shape[1] + xa
    else:
        xb = 2 * groupw * df_groups.shape[1] + xa
       
    dfc = df.copy()
    dfc.columns = pd.MultiIndex.from_frame(df_groups.reset_index())
    gr = dfc.groupby(level=list(range(len(df_groups.columns)+1))[1:], axis=1, sort=False)
    dfg = gr.mean().dropna(axis=1)
    
    origOrder = dfc.columns.droplevel('spot')[~dfc.columns.droplevel('spot').duplicated()]

    dfg = dfg.reindex(origOrder, axis=1)
    seg = gr.count().iloc[0].reindex(dfg.columns)
    
    # Add dendrogram obs
    if clusterObs and not keepOriginalOrderObs:  
        if clusterObsByGroups:
            if useMEDforObsGroups:
                print('Calculating sMED metric by groups.')
                print('Subsampling each group to have at most %s observations.' % sampleMED)
                dfP = pd.DataFrame(index=dfg.columns, columns=dfg.columns, data=0.)
                df_gt = df_groups.reset_index().set_index(dfP.index.names)

                for i_a in tqdm(range(len(dfP.index))):
                    item_a = dfP.index[i_a]
                    df_temp_a = df[df_gt.loc[item_a].values.T[0]] # vars by obs
                    df_temp_a = df_temp_a.sample(min(df_temp_a.shape[1], sampleMED), axis=1, replace=False)
                    for i_b in range(len(dfP.index)):
                        if i_b < i_a:
                            item_b = dfP.index[i_b]
                            df_temp_b = df[df_gt.loc[item_b].values.T[0]]
                            df_temp_b = df_temp_b.sample(min(df_temp_b.shape[1], sampleMED), axis=1, replace=False)
                            temp = cdist(df_temp_a.values.T, df_temp_b.values.T, metric=sMED).astype(int).mean()
                            dfP.loc[item_a, item_b] = temp
                            dfP.loc[item_b, item_a] = temp

                P = squareform(dfP.values)
            else:
                M = dfg.values.T
                M = np.nan_to_num(M, nan=np.max(M, axis=None))
                P = pdist(M, metric=linkageMetric)

            L = scipy.cluster.hierarchy.linkage(P, method=linkageMethod, optimal_ordering=False if useMEDforObsGroups else True)
            D = scipy.cluster.hierarchy.dendrogram(L, orientation='right', no_plot=True)

            orderObs = D['leaves']

            if returnDistancesOnly:
                dfP = pd.DataFrame(scipy.spatial.distance.squareform(P), index=dfg.columns, columns=dfg.columns)
                dfP = dfP.iloc[orderObs, orderObs]
                return dfP

            dfg = dfg.iloc[:, orderObs]

            tree = scipy.cluster.hierarchy.to_tree(L, rd=True)[1]
            meta = seg.rename('count').to_frame()
            for level in seg.index.names:
                meta[level + '_color'] = pd.Series(meta.index.get_level_values(level)).apply(lambda s: matplotlib.colors.to_hex(palette[s])).values

            meta = meta.reset_index()

            cond = meta['count'] > 0
            for key in reference.keys():
                cond = cond & (meta[key]==reference[key])
            try:
                root = np.where(cond)[0][0]
            except:
                print('Root not found')
                root = 0

            seg = seg.iloc[orderObs]
            #print(seg)

            ax = fig.add_axes([xc, ya, xd-xc-b, yb-ya-b], frame_on=False)

            ticks = seg.cumsum()-(0.5*seg).astype(int)
            orig_ticks = np.arange(5, len(D['leaves'])*10+5, 10).astype(float)
            for xs, ys in zip(np.array(D['icoord']), np.array(D['dcoord'])):
                xs = translate_pos(xs, ticks, orig_ticks)
                ax.plot(ys, xs, color=dendrogramLineColor, linewidth=dendrogramLineWidth, clip_on=False)
                
            ax.tick_params(length=0, labelsize=10, zorder=np.inf)
            #ax.set(xticks=[], xticklabels=[])
            #ax.set(yticks=ticks, yticklabels=orderObs)
            ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
            ax.set(ylim=(0.5, df.shape[1]+0.5))
            ax.set_facecolor('white')
            
        elif df.shape[1]<=safetyLimit:
            M = df.values.T
            M = np.nan_to_num(M, nan=np.max(M, axis=None))
            if useMEDforObs:
                print('Using sMED')
                P = pdist(M, metric=sMED)
            else:
                P = pdist(M, metric=linkageMetric)

            L = scipy.cluster.hierarchy.linkage(P, method=linkageMethod, optimal_ordering=optimalOrderingForObs)


            tree = scipy.cluster.hierarchy.to_tree(L, rd=True)[1]
            root = 0

            cols = df_groups.columns.values.tolist()
            meta = pd.Series(index=pd.MultiIndex.from_arrays(df_groups[cols].values.T, names=cols), data=1, name='count').to_frame()

            for level in cols:
                meta[level + '_color'] = pd.Series(meta.index.get_level_values(level)).apply(lambda s: matplotlib.colors.to_hex(palette[s])).values

            meta = meta.reset_index()


            ax = fig.add_axes([xc, ya, xd-xc-b, yb-ya-b], frame_on=False)
            
            origLineWidth = matplotlib.rcParams['lines.linewidth']
            matplotlib.rcParams['lines.linewidth'] = dendrogramLineWidth
            D = scipy.cluster.hierarchy.dendrogram(L, no_plot=False, color_threshold=0, above_threshold_color=dendrogramLineColor, get_leaves=True, orientation='right', ax=ax)
            matplotlib.rcParams['lines.linewidth'] = origLineWidth
            
            orderObs = df.columns[D['leaves']]
            
            ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
            ax.set_facecolor('white')
        else:
            print('Number of observation exceeds %s, keeping original order of observations.' % safetyLimit)
            print('To increase the limit change parameter safetyLimit.')
            print('Caution: Clustering more than a few thousand observations may be very slow.')
            keepOriginalOrderObs = True
            clusterObs = False
            
        
    if clusterVar and not keepOriginalOrderObs:
        # Add dendrogram var
        ax = fig.add_axes([xb, yb, xc-xb-b, ye-yb], frame_on=False)
        ax.grid(False)
        
        M = df.values
        linewidth = dendrogramLineWidth
        Z = scipy.cluster.hierarchy.linkage(np.nan_to_num(M, nan=np.max(M, axis=None)), method=linkageMethod, metric=linkageMetric, optimal_ordering=True)
        orderVar = scipy.cluster.hierarchy.dendrogram(Z, no_plot=True)['leaves']
        ax.set(xlim=(0, df.shape[0]))
        D = scipy.cluster.hierarchy.dendrogram(Z, p=p, truncate_mode='level', orientation='top', no_plot=True)
        
        if clusterVar:
            leg = pd.Series([eval(v) if '(' in v else 1 for v in D['ivl']])
            ticks = leg.cumsum()-(0.5*leg).astype(int)
            orig_ticks = np.arange(5, len(D['leaves'])*10+5, 10).astype(float)
            for xs, ys in zip(np.array(D['dcoord']), np.array(D['icoord'])):
                ys = translate_pos(ys, ticks, orig_ticks)
                ax.plot(np.array(ys)-0.5, xs, color=dendrogramLineColor, linewidth=dendrogramLineWidth, clip_on=False) 
                
        ax.set_facecolor('white')
        if clusterVar:
            for itick, tick in enumerate(ticks):
                l = leg.values[itick]/2
                gap = df.shape[0]/400
                arm = l if l-gap<gap else l-gap
                ax.plot([tick-arm-0.5, tick+arm-0.5], [0, 0], c=dendrogramLineColor, linewidth=5, clip_on=False, zorder=np.inf, solid_capstyle='butt')
                
        ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
        ax.set(xlim=(0, df.shape[0]), ylim=(0, ax.get_ylim()[1])) 
    
    # Add heatmap
    if True:
        ax = fig.add_axes([xb, ya, xc-xb-b, yb-ya-b], frame_on=True)
        ax.grid(False)
        
        if clusterVar and not keepOriginalOrderObs:
            dfc = dfc.iloc[orderVar, :]
            
        if type(dfg.columns) is pd.Index:
            dfg.columns = pd.MultiIndex.from_frame(dfg.columns.to_frame())
           
        df_reordered = pd.concat([dfc.xs(t, level=tuple(dfg.columns.names), axis=1, drop_level=False).sample(frac=1, replace=False, random_state=0, axis=1) for t in dfg.columns], axis=1)
            
        if keepOriginalOrderObs or (df_reordered.shape[1]>safetyLimit and not clusterObsByGroups):
            df_reordered = df_reordered[dfc.columns]
        elif not clusterObsByGroups:
            df_reordered = df_reordered.loc[:, orderObs]
        
        M = df_reordered.copy().values.T
        masked_M = np.ma.array(M, mask=np.isnan(M))
        cmap = copy.copy(colormap)
        cmap.set_bad(colormapBad)

        im = ax.imshow(masked_M, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, interpolation='None', extent=(0, M.shape[1], M.shape[0]-0.5, -0.5))
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.set(yticks=[], ylim=(-0.5, M.shape[0] - 0.5))
        if addGeneLabels:
            ax.set(xticks=np.array(list(range(M.shape[1])))+0.5)
            ax.set_xticklabels(dfc.index.values, rotation=90, fontsize=fontsizeGeneLabels)
        else:
            ax.set(xticks=[])

        ax.set(ylim=(-0.5, M.shape[0]-0.5))
        
        # TODO: make conditions to make sure the lines are off when observations aren't ordered by groups
        if addLinesOnHeatmap:
            for pos in [0] + seg.cumsum().values.tolist():
                ax.axhline(pos+0.5, color='k', linestyle='-', linewidth=1.0)
            
    # Add groups
    if True:
        if clusterVar:
            ax = fig.add_axes([xa, ya, xb-xa-b, yb-ya-b], frame_on=False)
        else:
            ax = fig.add_axes([(xb-xa)/2+xa, ya, (xb-xa)/2-b, yb-ya-b], frame_on=False)
            
        indexer = df_groups.index.reindex(df_reordered.columns.get_level_values(0))[1]
        if not indexer is None:
            df_groups_c = df_groups.iloc[df_groups.index.reindex(df_reordered.columns.get_level_values(0))[1]]
        else:
            df_groups_c = df_groups.copy()
        df_groups_co = df_groups_c.copy()
        
        ut = {matplotlib.colors.to_hex(palette[s]):s for s in df_groups_c.stack().unique()}
        
        for group in df_groups.columns:
            df_groups_c[group] = df_groups_c[group].apply(lambda s: matplotlib.colors.to_hex(palette[s]))
            
        ucolors = df_groups_c.stack().unique()
        dcolors = {ucolors[i]: i for i in range(len(ucolors))}
        
        for igroup, group in enumerate(df_groups.columns):
            df_groups_c[group] = df_groups_c[group].replace(dcolors)
            ax.plot([igroup, igroup], [0, len(df_groups)], c='w', linewidth=2., zorder=np.inf)
            
        ax.pcolormesh(df_groups_c.astype(float).values, cmap=matplotlib.colors.ListedColormap(dcolors))
        
        if addLinesOnGroups:
            for pos in [0] + seg.cumsum().values.tolist():
                ax.axhline(pos, color='k', linestyle='-', linewidth=1.0, clip_on=False)
           
        for igroup, group in enumerate(df_groups.columns):
            ax.text(igroup + 0.5, -0.01*max(ax.get_ylim()), group.title(), ha='center', va='top', fontsize=groupfontsize)
            
        ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[], ylim=(0, len(df_groups_c)))
 
        if len(reference)>0:
            try:
                levx = df_groups_co.shape[1]/2
                levy = df_groups_co.reset_index().set_index(list(reference.keys()))
                if len(reference.keys())==1:
                    levy.index = pd.MultiIndex.from_arrays([levy.index.values], names=[levy.index.name])
                levy['i'] = range(len(levy))
                for level in reference.keys():
                    levy = levy.xs(key=reference[level], level=level, drop_level=False)
                levy = levy['i'].values.mean()
                ltext = ax.text(levx, levy+0.5, referenceLabel, ha='center', va='center', fontsize=16, zorder=np.inf)
                ltext.set_path_effects([path_effects.Stroke(linewidth=3., foreground='w'), path_effects.Normal()])
            except Exception as exception:
                print('Could not add reference label')
                print(exception)
                    
        # Add groups legend  
        l = []
        for group in df_groups.columns:
            v = np.array(df_groups_c[group].drop_duplicates().sort_values().reset_index(drop=True).values)
            l.append(v)
            
        dfl = pd.DataFrame(l).T
        mval = dfl.stack().max() + 1
        dfl = dfl.fillna(mval).astype(int)
        dfl.columns = df_groups_c.columns
        dcolorsc = dcolors.copy()
        if (dfl.stack()==mval).sum()>0:
            dcolorsc.update({matplotlib.colors.to_hex(palette['-1']): mval})
            ut.update({matplotlib.colors.to_hex(palette['-1']):'-1'})
            
        dflc = dfl.applymap(lambda v: pd.Series(dcolorsc).reset_index().set_index(0)['index'].loc[v])
        dflc = dflc.applymap(lambda v: ut[v])
        
        for col in dflc.columns:
            wh = dflc[col].values!='-1'
            dfl.loc[wh, col] = dfl.loc[wh, col][np.argsort(dflc.loc[wh, col].values)].values
            
        vdiff = yd-yb - (yd-yb)*dfl.shape[0]/16
        if clusterVar:
            ax = fig.add_axes([xa, yb+vdiff, xb-xa-b, yd-yb-vdiff], frame_on=False)
        else:
            ax = fig.add_axes([xa, yb-ye+yb+vdiff, (xb-xa)/2-b, yd-yb-vdiff], frame_on=False)
        
        ax.pcolormesh(dfl.values, cmap=matplotlib.colors.ListedColormap(dcolorsc), edgecolors='w')
        ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[], ylim=(len(dfl), 0))
        
        for igroup, group in enumerate(df_groups.columns):
            dnames = pd.DataFrame(pd.concat([df_groups_c[group], df_groups[group]], axis=1).drop_duplicates().values).set_index(0)[1].to_dict()
            for ientry, entry in enumerate(dfl[group]):
                if entry in dnames.keys():
                    ltext = ax.text(igroup + 0.5, ientry + 0.5, dnames[entry], ha='center', va='center', fontsize=groupfontsize)
                    ltext.set_path_effects([path_effects.Stroke(linewidth=3., foreground='w'), path_effects.Normal()])
             
        for igroup, group in enumerate(df_groups.columns):
            ax.text(igroup + 0.5, -1.25 + 0.5, group.title(), ha='center', va='center', fontsize=groupfontsize)
        
    # Add colorbar
    if True:
        if clusterObs:
            ax = fig.add_axes([xd+b, ya, xe-xd, yc-ya], frame_on=False)
        else:
            ax = fig.add_axes([xc+b, ya, xe-xd, yc-ya], frame_on=False)
        ax.grid(False)
        ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
        clb = fig.colorbar(im, ax=ax, fraction=1.0, label=colorbarLabel)
        mmax = df_reordered.max().max() if vmax is None else vmax
        mmin = df_reordered.min().min() if vmin is None else vmin
        
        clb.ax.set_yticks([mmax, mmin])
        if not colorbarLabels is None:
            clb.ax.set_yticklabels(colorbarLabels)
        else:
            clb.ax.set_yticklabels([np.round(mmax, 1), np.round(mmin, 1)])
        clb.ax.tick_params(length=0, labelsize=12)
        ax.axis('off')
       
    df_reordered.columns = pd.MultiIndex.from_frame(df_groups_co.reset_index())
    df_reordered = df_reordered.T
    
    if saveFig:
        plt.savefig(figureName, dpi=dpi)
    
    if useMEDforObsGroups and clusterObsByGroups:
        return root, tree, meta
    else:
        return df_reordered.reindex(index=df_reordered.index[::-1])


def prepForCplot(ad_all, sample_name='sample', cluster_name='cluster', quantile=0.95, cap=None):

    def getDFFromAD(ad, sample_name=sample_name, cluster_name=cluster_name):
        df_temp = ad.to_df().T
        df_temp.columns = pd.MultiIndex.from_frame(ad.obs.reset_index()[['index', sample_name, cluster_name]], names=['spot', 'sample', 'cluster'])
        df_temp.index = pd.Index(ad.var.reset_index()['index'], name='symbol')
        df_temp = df_temp.loc[(df_temp>0).mean(axis=1)>0]
        return df_temp

    df_all = getDFFromAD(ad_all)
    df_meta = df_all.columns.to_frame()
    df_meta = df_meta.droplevel(['sample', 'cluster'], axis=0)[['sample', 'cluster']]
    df_meta['time'] = df_meta['sample'].str.split('_', expand=True)[1]
    df_meta['sample'] = df_meta['sample'].apply(lambda s: s.split('_')[1] + ' ' + s.split('_')[2])
    df_meta = df_meta[['sample', 'time', 'cluster']].astype('category')
    df_all = df_all.droplevel(['sample', 'cluster'], axis=1)
    print(df_all.shape)

    df_all = (df_all.T / np.nanquantile(df_all.replace(0, np.nan), quantile, axis=1)).T.dropna()

    if not cap is None:
        df_all[df_all > cap] = cap

    return df_all, df_meta