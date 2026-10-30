import sys,json,numpy as np
from feats import *
from common import fit_pred, StratifiedGroupKFold
from sklearn.model_selection import StratifiedKFold
V={'default (SG11, poly5, z)':dict(order=5,sg=11,norm='z'),'poly3':dict(order=3,sg=11,norm='z'),'poly7':dict(order=7,sg=11,norm='z'),
   'no smoothing':dict(order=5,sg=0,norm='z'),'vector norm':dict(order=5,sg=11,norm='l2'),'area norm':dict(order=5,sg=11,norm='area'),'1445 norm':dict(order=5,sg=11,norm='1445')}
cfg={'four':dict(k=18,C=10,g=1),'binary':dict(k=44,C=10,g=1)}
out={}
for name,v in V.items():
    s,Z=prep_fp(**v)
    for task in ['four','binary']:
        y=y4 if task=='four' else y2
        for unit in ['spectrum','donor']:
            acc=[];ilc=[]
            for sd in range(3):
                cv=StratifiedKFold(20,shuffle=True,random_state=sd) if unit=='spectrum' else StratifiedGroupKFold(20,shuffle=True,random_state=sd)
                p=np.zeros(len(y),int)
                for tr,te in (cv.split(Z,y) if unit=='spectrum' else cv.split(Z,y,donor)): p[te]=fit_pred('svm',cfg[task],Z[tr],y[tr],Z[te])
                acc.append(np.mean(p==y)); ilc.append(np.mean(p[y==2]==2) if task=='four' else np.nan)
            out[f'{name}|{task}|{unit}']=dict(acc=float(np.mean(acc)),sd=float(np.std(acc)),ilc=float(np.mean(ilc)))
            print(name,task,unit,round(np.mean(acc)*100,1),round(np.nanmean(ilc)*100,1),flush=True)
    json.dump(out,open('prepsens.json','w'),indent=1)
