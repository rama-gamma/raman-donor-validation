# Split-conformal with donor-disjoint calibration; spectrum-level and sample-level (aggregated) variants
import json,numpy as np
from common import *
from sklearn.svm import SVC
def svm_proba(Xtr,ytr,Xs,p,seed):
    pc=PCA(p['k']).fit(Xtr); Z=pc.transform(Xtr); g=p['g']/(Z.shape[1]*Z.var())
    m=SVC(C=p['C'],gamma=g,probability=True,random_state=seed).fit(Z,ytr)
    return [m.predict_proba(pc.transform(X)) for X in Xs]
def qhat(scores,alpha):
    n=len(scores); q=np.ceil((n+1)*(1-alpha))/n
    return np.quantile(scores,min(q,1),method='higher')
cfg={'four':dict(k=18,C=10,g=1),'binary':dict(k=44,C=10,g=1)}
R={}
for task in ['binary','four']:
    y=y4 if task=='four' else y2; nc=len(np.unique(y)); rows=[]
    for s in range(10):
        rng=np.random.default_rng(s)
        S_cov=np.zeros(len(y)); S_size=np.zeros(len(y)); S_mcov=np.zeros(len(y)); S_msize=np.zeros(len(y))
        smp_res=[]
        for tr,te in StratifiedGroupKFold(10,shuffle=True,random_state=s).split(Y,y,donor):
            dn=np.unique(donor[tr]); cal_d=rng.choice(dn,int(round(0.25*len(dn))),replace=False)
            cal=tr[np.isin(donor[tr],cal_d)]; fit=tr[~np.isin(donor[tr],cal_d)]
            Pc,Pt=svm_proba(Y[fit],y[fit],[Y[cal],Y[te]],cfg[task],s)
            sc=1-Pc[np.arange(len(cal)),y[cal]]
            q=qhat(sc,0.1); sets=(1-Pt)<=q
            S_cov[te]=sets[np.arange(len(te)),y[te]]; S_size[te]=sets.sum(1)
            # Mondrian (class-conditional)
            ms=np.zeros_like(sets)
            for c in range(nc):
                scc=sc[y[cal]==c]; qc=qhat(scc,0.1) if len(scc)>0 else 1
                ms[:,c]=(1-Pt[:,c])<=qc
            S_mcov[te]=ms[np.arange(len(te)),y[te]]; S_msize[te]=ms.sum(1)
            # sample-level: average probabilities over spectra of a sample
            def agg(idx,P):
                us=np.unique(sample[idx]); A=np.array([P[sample[idx]==u].mean(0) for u in us]); ly=np.array([y[idx][sample[idx]==u][0] for u in us]); return A,ly
            Ac,lc=agg(cal,Pc); At,lt=agg(te,Pt)
            q2=qhat(1-Ac[np.arange(len(lc)),lc],0.1); st=(1-At)<=q2
            for i in range(len(lt)): smp_res.append((st[i,lt[i]],st[i].sum(),lt[i],len(lc)))
        single=S_size==1
        sm=np.array(smp_res)
        rows.append(dict(cov=S_cov.mean(),single=single.mean(),acc_single=float(np.mean(S_cov[single])) if single.any() else np.nan,size=S_size.mean(),
            cov_class=[S_cov[y==c].mean() for c in range(nc)],
            mcov=S_mcov.mean(),msize=S_msize.mean(),mcov_class=[S_mcov[y==c].mean() for c in range(nc)],
            s_cov=sm[:,0].mean(),s_size=sm[:,1].mean(),s_single=(sm[:,1]==1).mean(),s_ncal=sm[:,3].mean(),
            s_cov_class=[sm[sm[:,2]==c,0].mean() for c in range(nc)]))
        print(task,s,{k:np.round(v,3) for k,v in rows[-1].items()},flush=True)
    R[task]={k:np.nanmean([r[k] for r in rows],0).tolist() for k in rows[0]}
    R[task].update({k+'_sd':np.nanstd([r[k] for r in rows],0).tolist() for k in ['cov','single','size','s_cov']})
json.dump(R,open('conformal.json','w'),indent=1)
