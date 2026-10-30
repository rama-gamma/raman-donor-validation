# Same pipeline, four splitting units: spectrum, measurement point, sample, donor (20-fold, PCA inside folds)
import json,sys,numpy as np
from common import *
from sklearn.model_selection import StratifiedKFold
out={}
cfg={'four':dict(k=18,C=10,g=1),'binary':dict(k=44,C=10,g=1)}
for task in ['binary','four']:
    y=y4 if task=='four' else y2
    for unit,g in [('spectrum',None),('point',point),('sample',sample),('donor',donor)]:
        accs=[]; sens=[]
        for s in range(10):
            cv=StratifiedKFold(20,shuffle=True,random_state=s) if g is None else StratifiedGroupKFold(20,shuffle=True,random_state=s)
            pred=np.zeros(len(y),int)
            for tr,te in (cv.split(Y,y) if g is None else cv.split(Y,y,g)):
                pred[te]=fit_pred('svm',cfg[task],Y[tr],y[tr],Y[te])
            accs.append(np.mean(pred==y)); sens.append([np.mean(pred[y==c]==c) for c in np.unique(y)])
        out[f'{task}_{unit}']=dict(mean=float(np.mean(accs)),sd=float(np.std(accs)),sens=np.mean(sens,0).tolist())
        print(task,unit,round(np.mean(accs)*100,1),round(np.std(accs)*100,1),np.round(np.mean(sens,0)*100,1),flush=True)
json.dump(out,open('split_levels.json','w'),indent=1)
