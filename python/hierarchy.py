# Similarity of spectra by level: same point (repeats), same sample different point, same class different donor
import json,numpy as np
from common import *
C=np.corrcoef(Y); iu=np.triu_indices(len(Y),1)
sp=point[:,None]==point[None]; ss=sample[:,None]==sample[None]; sd=donor[:,None]==donor[None]; sc=y4[:,None]==y4[None]
lev={'same point (repeat)':sp,'same sample, other point':ss&~sp,'same class, other donor':sc&~sd}
out={k:float(np.median(C[iu][m[iu]])) for k,m in lev.items()}
reps=[np.sum(point==p) for p in np.unique(point)]
out['points']=int(len(np.unique(point))); out['spectra_per_point_mean']=float(np.mean(reps))
print(out); json.dump(out,open('hierarchy.json','w'),indent=1)
