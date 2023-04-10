import EoN
from EoN.auxiliary import hierarchy_pos
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def plotTree(root, nodes, meta, seed=None, useSpringLayout=False, figsize=(40,10)):
    
    np.random.seed(seed)

    cols = meta.columns.difference(['count'])
    cols = cols[~cols.str.contains('_color')].values.tolist()

    G = nx.Graph()
    for i in range(len(nodes)):
        node = nodes[i]
        id = node.get_id()
        size = meta.loc[id]['count'] if id < len(meta) else 1
        size += 25

        label = '-'.join([meta.loc[id][col] for col in cols]) if id < len(meta) else ''
        #label = id
        cluster_color = meta.loc[id]['cluster_color'] if id < len(meta) else 'k'
        time_color = meta.loc[id]['time_color'] if id < len(meta) else 'k'

        G.add_nodes_from([(id, {'cluster_color': cluster_color, 'time_color': time_color, 'weight': size, 'label': label}), ])
        if not node.is_leaf():
            for child in [node.get_left(), node.get_right()]:
                cid = child.get_id()
                edge_length = max(node.dist-child.dist, 0.1)
                edge_color = meta.loc[cid]['time_color'] if cid < len(meta) else 'grey'
                edge_width = 7 if cid < len(meta) else 7
                G.add_edge(node.get_id(), child.get_id(), weight=1/edge_length, color=edge_color, width=edge_width)

    if useSpringLayout:
        pos=nx.spring_layout(G)
    else:
        pos=hierarchy_pos(G, root)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(G, pos=pos, with_labels=False, alpha=1.0,
            nodelist=[],
            edge_color=['grey' for i in G.edges()],
            width=[8 for i in G.edges()],
            ax=ax)
    nx.draw(G, pos=pos, with_labels=True, alpha=1.0,
            node_color=[G.nodes[i]['cluster_color'] for i in G.nodes()],
            edgecolors=['grey' for i in G.nodes()],
            linewidths=1,
            node_size=[G.nodes[i]['weight'] for i in G.nodes()],
            labels={i: G.nodes[i]['label'] for i in G.nodes()},
            edge_color=[G.edges[i]['color'] for i in G.edges()],
            width=[G.edges[i]['width'] for i in G.edges()],
            ax=ax)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=[root], node_color='grey', node_size=1000, ax=ax)
    plt.show()
    return