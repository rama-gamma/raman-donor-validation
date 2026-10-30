import json,numpy as np
from feats import *
out={}
for model in ['lr','svm']:
    R=json.load(open(f'regions_{model}.json'))
    for key,preds in R.items():
        fs,task=key.split('|'); y=y4 if task=='four' else y2
        P=np.array(preds); acc=(P==y).mean(1)
        sens=np.array([[np.mean(p[y==c]==c) for c in np.unique(y)] for p in P]).mean(0)
        # donor bootstrap CI on pooled predictions
        rng=np.random.default_rng(0); D=np.unique(donor); idx={d_:np.where(donor==d_)[0] for d_ in D}; bs=[]
        for _ in range(2000):
            ii=np.concatenate([idx[d_] for d_ in rng.choice(D,len(D))]); bs.append((P[:,ii]==y[ii]).mean())
        out[f'{model}|{fs}|{task}']=dict(acc=float(acc.mean()),sd=float(acc.std()),bal=float(sens.mean()),sens=sens.tolist(),ci=np.percentile(bs,[2.5,97.5]).tolist())
        o=out[f'{model}|{fs}|{task}']; print(model,fs.ljust(7),task.ljust(6),round(o['acc']*100,1),'±',round(o['sd']*100,1),'bal',round(o['bal']*100,1),np.round(sens*100).astype(int),np.round(o['ci'],3)*100)
json.dump(out,open('regions_summary.json','w'),indent=1)
