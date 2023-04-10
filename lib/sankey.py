
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline
from plotly.offline import plot_mpl
# Image(url="%s.png" % nameAppend, width=400)

def alignSeries(se1, se2, tagForMissing):
    se1.index.name = 'index'
    se2.index.name = 'index'
    append = lambda se1, se2: pd.concat([se1, pd.Series(index=se2.index.difference(se1.index), data=[tagForMissing] * len(se2.index.difference(se1.index)), dtype=object)], axis=0, sort=False)
    se1 = append(se1, se2)
    se2 = append(se2, se1)
    se1.name = 'se1'
    se2.name = 'se2'
    return pd.concat((se1, se2.loc[se1.index]), axis=1, sort=True)

def getCountsDataframe(se1, se2, tagForMissing='N/A'):
    df = alignSeries(se1, se2, tagForMissing)
    counts = {group[0]:{k:len(v) for k, v in group[1].groupby(by='se1').groups.items()} for group in df.reset_index(drop=True).set_index('se2').groupby('se2')}
    df = pd.DataFrame.from_dict(counts).fillna(0.0).astype(int)
    moveTag = lambda df: pd.concat([df.iloc[np.where(df.index != tagForMissing)[0]], df.iloc[np.where(df.index == tagForMissing)[0]]], axis=0, sort=False) if tagForMissing in df.index else df
    return moveTag(moveTag(df.T).T)



def makeSankeyDiagram(df, colormapForIndex = None, colormapForColumns = None, linksColor = 'rgba(100,100,100,0.6)', title = '', attemptSavingHTML = False, quality = 4,
                      width = 400, height = 400, border = 20, nodeLabelsFontSize = 15, nameAppend = '_Sankey_diagram', saveDir=''):

    try:
        temp_index = pd.MultiIndex.from_arrays([df.index, [colormapForIndex[item] for item in df.index]], names=['label', 'color'])
        temp_columns = pd.MultiIndex.from_arrays([df.columns, [colormapForColumns[item] for item in df.columns]], names=['label', 'color'])
        df.index = temp_index
        df.columns = temp_columns
    except Exception as exception:
        #print('Using default node colors')
        colormapForIndex = None
        colormapForColumns = None

    if (colormapForIndex is None) or (colormapForColumns is None):
        nodeColors = ['rgba(150,0,10,0.8)'] * len(df.index) + ['rgba(10,0,150,0.8)'] * len(df.columns)
        nodeLabels = df.index.to_list() + df.columns.to_list()
    else:
        nodeLabels = df.index.get_level_values('label').to_list() + df.columns.get_level_values('label').to_list()
        nodeColors = df.index.get_level_values('color').to_list() + df.columns.get_level_values('color').to_list()

    sources, targets, values, labels = [], [], [], []
    for i, item in enumerate(df.index):
        sources.extend([i] * len(df.loc[item]))
        targets.extend(list(range(len(df.index), len(df.index) + len(df.loc[item]))))
        values.extend([j for j in df.loc[item].values])
        if type(item) is tuple:
            labels.extend([str(item[0]) + ' -> ' + str(jtem[0]) for jtem in df.loc[item].index])
        else:
            labels.extend([str(item) + ' -> ' + str(jtem) for jtem in df.loc[item].index])

    colorscales = [dict(label=label, colorscale=[[0, linksColor], [1, linksColor]]) for label in labels]

    if not nodeColors is None:
        for i in range(len(sources)):
            if nodeColors[sources[i]] == nodeColors[targets[i]]:
                newColor = ','.join(nodeColors[sources[i]].split(',')[:3] + ['0.6)'])
                colorscales[i] = dict(label=labels[i], colorscale=[[0, newColor], [1, newColor]])

    fig = go.Figure(data=[go.Sankey(valueformat = '', valuesuffix = '', textfont = dict(color = 'rgb(0,0,0)', size = nodeLabelsFontSize, family = 'Arial'),
        node = dict(pad = 20, thickness = 40, line = dict(color = 'white', width = 0.0), label = nodeLabels, color = nodeColors,), # hoverlabel=dict(bordercolor = 'yellow')
        link = dict(source = sources, target = targets, value = values, label = labels, colorscales = colorscales, hoverinfo='all'),)],) #line ={'color':'rgba(255,0,0,0.8)', 'width':0.1}

    if not title is None:
        fig.update_layout(title_text=title, font_size=10)

    fig.update_layout(margin=dict(l=border, r=border, t=border, b=border))

    try:
        fig.write_image(os.path.join(saveDir, nameAppend + '.png'), width=width, height=height, scale=quality)
    except Exception as exception:
        print('Cannot save static image (likely due to missing orca). Saving to interactive html')

    return fig

from sklearn.cluster import SpectralCoclustering
def reorder(df, n_co_clusters=4):
    model = SpectralCoclustering(n_clusters=n_co_clusters, random_state=0)
    model.fit(df.values)
    rows_order, cols_order = np.argsort(model.row_labels_), np.argsort(model.column_labels_)
    df = pd.DataFrame(data=df.values[rows_order][:, cols_order], index=df.index[rows_order], columns=df.columns[cols_order])
    return df

# makeSankeyDiagram(getCountsDataframe(se1, se2), nameAppend='_Q', border=25)
# makeSankeyDiagram(getCountsDataframe(ads[ids[0]].obs['all_inc_clusters_0.15'], ads[ids[0]].obs['inception_clusters_seurat']), nameAppend='_Q', border=25);