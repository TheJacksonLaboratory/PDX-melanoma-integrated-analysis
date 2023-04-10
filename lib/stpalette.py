
import pandas as pd
import matplotlib
from matplotlib import cm

palette1={str(k):cm.Set1(k) for k in range(8)}
palette1.update({str(k):cm.Set2(k-8) for k in range(8,16)})

palette1.update({str(k)+'T4':cm.Set1(k) for k in range(8)})
palette1.update({str(k)+'T4':cm.Set2(k-8) for k in range(8,16)})

palette1.update({'mouse':cm.Set1(5), 'human':cm.Set1(1), 'both':cm.Set1(0), 'low':'grey'})
palette1.update({str(3):cm.Set1(5), str(5):cm.Set1(3), str(8):cm.Set2(0), str(7):'cyan', str(6):cm.Set1(7)})
palette1.update({str(3)+'T4':cm.Set1(5), str(5)+'T4':cm.Set1(3), str(8)+'T4':cm.Set2(0), str(7)+'T4':'cyan', str(6)+'T4':cm.Set1(7)})

palette1.update({'TC': 'dodgerblue', 'T0': 'navy', 'TE': 'indigo', 'T1': 'green', 'T2': 'yellow', 'T3': 'orange', 'T4': 'red'})

palette1.update({'Treated': 'crimson', 'Untreated': 'green'})

palette1.update({'-1':'white', '9':'k'})


palette_WM4007_rna = palette1.copy()
palette_WM4007_rna.update({'4':cm.Set1(5), '7':'cyan', '3':cm.Set1(4), '2':cm.Set1(2), '8':cm.Set2(0), '0':cm.Set1(0), '5':cm.Set1(3), '6':cm.Set1(7), '1':cm.Set1(1), '9':'black'})

palette_WM4237_rna = palette1.copy()
palette_WM4237_rna.update({'7':cm.Set1(5), '3':'cyan', '4':cm.Set1(4), '8':cm.Set1(2), '0':cm.Set1(0), '6':cm.Set1(3), '5':cm.Set1(7), '1':cm.Set1(1), '2':'olive'})

def writePalette(palette, label='general'):
    palette1hex = pd.DataFrame([[k, matplotlib.colors.to_rgba(palette[k]), matplotlib.colors.to_hex(palette[k])] for k in palette.keys()])
    palette1hex.columns = ['identity', 'color_python', 'color_R']
    palette1hex.to_csv(f'palette_{label}.csv', index=False)
    return palette1hex

if False:
    writePalette(palette_WM4237_rna, label='WM4237_rna')
    writePalette(palette_WM4007_rna, label='WM4007_rna')
    writePalette(palette1, label='general')
