# Physics-based simulation: raw counts = gain * (chemical spectrum) + fluorescence background, Poisson shot noise,
# then the full preprocessing chain (SG, iterative polynomial baseline, z-score) and the fixed PCA-SVM pipeline.
import sys,json,numpy as np
from prep import baseline
from feats import *
from scipy.signal import savgol_filter
from pcaopt import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
b=np.load('bc.npz'); sF,Fbc=b['sF'],b['Fbc']
m=(shift>=800)&(shift<=2000); RAWs=savgol_filter(X[:,m],11,3,axis=1,mode='interp')
BG=np.clip(RAWs-Fbc,0,None)                       # bank of real fluorescence backgrounds
unit=np.array([f'{d_}_{h}' for d_,h in zip(donor,hist)]); U=np.unique(unit)
mu=np.array([Fbc[y4==c].mean(0) for c in range(4)])
Rr=Fbc-mu[y4]; Dm=np.array([Rr[unit==u].mean(0) for u in U]); W=Rr-np.array([Dm[list(U).index(u)] for u in unit])
nbar=len(Fbc)/len(U)
pb=PCA(20).fit(Dm); pw=PCA(20).fit(W)
sw_in_b=np.var(pw.transform(W)@pw.components_@pb.components_.T,0)  # within variance projected on between axes
vb=np.maximum(pb.explained_variance_-sw_in_b/nbar,0)
vw=pw.explained_variance_
g_sd=np.std(np.log(np.array([RAWs[unit==u].mean() for u in U])))*0.5
def chem(rng,c,donor_comp):
    return mu[c]+ (donor_comp if donor_comp is not None else 0)
def cohort(rng,sc,q,nd=(46,14,9,11),per=9):
    S=[];yy=[];gg=[];gid=0
    for c,n in enumerate(nd):
        for _ in range(n):
            dc=(rng.normal(0,np.sqrt(vb))@pb.components_) if sc['comp'] else None
            gd=np.exp(rng.normal(0,g_sd)) if sc['gain'] else None
            bgd=BG[rng.integers(len(BG))]*np.exp(rng.normal(0,.3)) if sc['bg'] else None
            for _ in range(per):
                x=mu[c]+(dc if dc is not None else (rng.normal(0,np.sqrt(vb))@pb.components_))
                x=x+rng.normal(0,np.sqrt(vw))@pw.components_
                g=gd if gd is not None else np.exp(rng.normal(0,g_sd))
                bg=(bgd if bgd is not None else BG[rng.integers(len(BG))]*np.exp(rng.normal(0,.3)))*np.exp(rng.normal(0,.1))
                lam=np.clip(g*x+bg,0,None)*q
                S.append(rng.poisson(lam)/q); yy.append(c); gg.append(gid)
            gid+=1
    S=np.array(S,float); S=savgol_filter(S,11,3,axis=1,mode='interp')
    S=np.array([baseline(sF,s_,5) for s_ in S]); S=zs(S)
    return S,np.array(yy),np.array(gg)
clf=lambda: make_pipeline(PCA(18),SVC(C=10,gamma='scale'))
def cvacc(X,y,cv,g=None):
    p=np.zeros(len(y),int)
    for tr,te in (cv.split(X,y) if g is None else cv.split(X,y,g)): p[te]=clf().fit(X[tr],y[tr]).predict(X[te])
    return np.mean(p==y)
SC={'shot noise only':dict(comp=0,gain=0,bg=0),'donor background':dict(comp=0,gain=0,bg=1),'donor gain':dict(comp=0,gain=1,bg=0),
    'donor chemistry':dict(comp=1,gain=0,bg=0),'all donor effects':dict(comp=1,gain=1,bg=1)}
if __name__=='__main__':
    nrep=int(sys.argv[1]); out={}
    runs=[(k,1.0) for k in SC]+[('all donor effects',0.1),('all donor effects',10.0)]
    for name,q in runs:
        bs=[];bd=[];tr_=[]
        for r in range(nrep):
            rng=np.random.default_rng(7000+r)
            X1,y1,g1=cohort(rng,SC[name],q); X2,y2_,_=cohort(rng,SC[name],q)
            t=np.mean(clf().fit(X1,y1).predict(X2)==y2_)
            bs.append(cvacc(X1,y1,StratifiedKFold(10,shuffle=True,random_state=r))-t)
            bd.append(cvacc(X1,y1,StratifiedGroupKFold(10,shuffle=True,random_state=r),g1)-t); tr_.append(t)
        key=f'{name}|q={q}'
        out[key]=dict(bias_spec=float(np.mean(bs)),se_spec=float(np.std(bs)/np.sqrt(nrep)),bias_don=float(np.mean(bd)),se_don=float(np.std(bd)/np.sqrt(nrep)),true=float(np.mean(tr_)))
        print(key,{a:round(v*100,1) for a,v in out[key].items()},flush=True)
        json.dump(out,open('sim_phys.json','w'),indent=1)
