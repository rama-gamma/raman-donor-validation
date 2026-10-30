import sys,json,time,numpy as np
import common
common.KMAX=60
from common import Fold, StratifiedGroupKFold
from feats import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
b=np.load('bc.npz'); FP=common.Y; HW=zs(b['Hbc'])
COMB=np.hstack([FP/np.sqrt(FP.shape[1]),HW/np.sqrt(HW.shape[1])])
RAT=np.load('ratios.npy')
SETS={'FP':FP,'HW':HW,'FP+HW':COMB,'ratios':RAT}
K=[3,5,8,10,15,20,30,44,60]
def grid(model,fs):
    if fs=='ratios':
        if model=='lr': return [dict(C=C) for C in [0.01,0.1,1,10,100]]
        return [dict(C=C,g=g) for C in [1,10,100] for g in [0.3,1,3]]
    if model=='lr': return [dict(k=k,C=C) for k in K for C in [0.01,0.1,1,10]]
    return [dict(k=k,C=C,g=g) for k in [5,10,20,44,60] for C in [1,10,100] for g in [0.3,1,3]]
class RFold:
    def __init__(s,model,Xtr,ytr,Xte): s.model=model; sc=StandardScaler().fit(Xtr); s.Ztr=sc.transform(Xtr); s.Zte=sc.transform(Xte); s.ytr=ytr
    def predict(s,p):
        m=LogisticRegression(C=p['C'],max_iter=3000) if s.model=='lr' else SVC(C=p['C'],gamma=p['g']/s.Ztr.shape[1])
        return m.fit(s.Ztr,s.ytr).predict(s.Zte)
def mk(model,fs,Xtr,ytr,Xte):
    if fs=='ratios': return RFold(model,Xtr,ytr,Xte)
    return Fold(model,Xtr,ytr,Xte)
def run(model,fs,task,nseed):
    X=SETS[fs]; y=y4 if task=='four' else y2; G=grid(model,fs); res=[]
    for s in range(nseed):
        pred=np.full(len(y),-1)
        for tr,te in StratifiedGroupKFold(10,shuffle=True,random_state=s).split(X,y,donor):
            sc=np.zeros(len(G))
            for itr,ite in StratifiedGroupKFold(4,shuffle=True,random_state=100+s).split(X[tr],y[tr],donor[tr]):
                F=mk(model,fs,X[tr][itr],y[tr][itr],X[tr][ite])
                for j,p in enumerate(G): sc[j]+=np.mean(F.predict(p)==y[tr][ite])
            pred[te]=mk(model,fs,X[tr],y[tr],X[te]).predict(G[int(sc.argmax())])
        res.append(pred.tolist())
        print(model,fs,task,s,round(np.mean(pred==y),3),flush=True)
    return res
if __name__=='__main__':
    model=sys.argv[1]; nseed=int(sys.argv[2]); out={}
    for task in ['four','binary']:
        for fs in ['HW','FP+HW','ratios','FP']:
            out[f'{fs}|{task}']=run(model,fs,task,nseed)
            json.dump(out,open(f'regions_{model}.json','w'))
