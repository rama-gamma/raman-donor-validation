import sys,json,numpy as np
import common
from common import fit_pred, StratifiedGroupKFold
from feats import *
FP=common.Y
cfg={'four':dict(k=18,C=10,g=1),'binary':dict(k=44,C=10,g=1)}
out={}
for task in ['four','binary']:
    y=y4 if task=='four' else y2
    for f in [0.25,0.5,0.75,1.0]:
        R=[]
        for rep in range(10):
            rng=np.random.default_rng(rep); p=np.full(len(y),-1); ntr=[]
            for tr,te in StratifiedGroupKFold(10,shuffle=True,random_state=rep).split(FP,y,donor):
                keep=[]
                for c in range(4):
                    ds=np.unique(donor[tr][y4[tr]==c]); n=max(1,int(round(f*len(ds))))
                    keep+=list(rng.choice(ds,n,replace=False))
                sel=tr[np.isin(donor[tr],keep)]; ntr.append([len(np.unique(donor[sel][y4[sel]==c])) for c in range(4)])
                p[te]=fit_pred('svm',cfg[task],FP[sel],y[sel],FP[te])
            sens=[np.mean(p[y==c]==c) for c in np.unique(y)]
            R.append(dict(acc=np.mean(p==y),bal=np.mean(sens),sens=sens,ntr=np.mean(ntr,0).tolist()))
        out[f'{task}|{f}']=dict(acc=float(np.mean([r['acc'] for r in R])),acc_sd=float(np.std([r['acc'] for r in R])),bal=float(np.mean([r['bal'] for r in R])),
            sens=np.mean([r['sens'] for r in R],0).tolist(),ntr=np.mean([r['ntr'] for r in R],0).tolist())
        print(task,f,{k:(np.round(v,3) if not isinstance(v,float) else round(v,3)) for k,v in out[f'{task}|{f}'].items()},flush=True)
json.dump(out,open('lcurve.json','w'),indent=1)
