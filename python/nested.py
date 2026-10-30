import sys, json, numpy as np, time
from common import *
name=sys.argv[1]; task=sys.argv[2]; nseed=int(sys.argv[3])
y=y4 if task=='four' else y2
res=[]
for s in range(nseed):
    t0=time.time(); pred=np.full(len(y),-1); chosen=[]
    for tr,te in StratifiedGroupKFold(10,shuffle=True,random_state=s).split(Y,y,donor):
        b=select(name,Y[tr],y[tr],donor[tr],s); chosen.append(b)
        pred[te]=fit_pred(name,b,Y[tr],y[tr],Y[te])
    res.append(dict(seed=s,pred=pred.tolist(),chosen=chosen))
    print(name,task,s,round(np.mean(pred==y),4),round(time.time()-t0),flush=True)
    json.dump(res,open(f'nested_{name}_{task}.json','w'))
