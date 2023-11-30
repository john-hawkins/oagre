import pandas as pd
import numpy as np
df = pd.read_csv("results/results_summary_file_full_new.csv")

import matplotlib.pyplot as plt

def find_winner(row):
    labels = ['GBM','LGBM','XTR','DTR','OAGRE-XT','OAGRE-DT' ]
    scores = [ row['GBM'], row['LGBM'], row['XTR'], row['DTR'], row['OAGRE-XT'], row['OAGRE-DT'] ]
    index = np.array(scores).argmin()
    diffs = [ 100*(scores[x] - scores[index])/scores[x] for x in range(len(scores)) if x != index]
    return labels[index], min(diffs)

df[['winner', 'lead']] = df.apply(find_winner, axis=1, result_type="expand")

oagred = df[(df['winner']=='OAGRE-XT') | (df['winner']=='OAGRE-DT')]

df['oagred'] = np.where((df['winner']=='OAGRE-XT') | (df['winner']=='OAGRE-DT'), 1.0, 0.0)
 
df['expression_nonline_prop_simple'] = df['expression_nonline_prop'].round(1)
temp = df.groupby('expression_nonline_prop_simple')['oagred'].mean().reset_index()

plt.scatter(temp['expression_nonline_prop_simple'], temp['oagred'], color='blue')
plt.ylim(0.0, 1.0)
plt.xlabel('Proportion of non-linear elements in function')
plt.ylabel('Proportion of datasets won by OAGRE')
plt.savefig("results/wins_vs_nonlinear_prop.png")

df['winner'].value_counts() 

df[df['winner']=='OAGRE-XT'].loc[:,['dataset_size','param_size', 'outlier_prop', 'winner', 'expression_nonline_prop']]
