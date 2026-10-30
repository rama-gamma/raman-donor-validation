# Simulation in PC space with between/within-donor variance estimated from real data
import json,numpy as np
from common import *
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
Z=PCA(20).fit_transform(Y)
mu=np.array([Z[y4==c].mean(0) for c in range(4)])
# variance components per PC, pooled over classes (one-way random effects, donor within class)
R=Z-mu[y4]
dm={d_:R[donor==d_].mean(0) for d_ in np.unique(donor)}
W=np.array([R[i]-dm[donor[i]] for i in range(len(R))]); sw2=W.var(0,ddof=1)
nbar=np.mean([np.sum(donor==d_) for d_ in dm]); B=np.array(list(dm.values()))
sb2=np.maximum(B.var(0,ddof=1)-sw2/nbar,0)
share=sb2/(sb2+sw2)
print('donor share mean',share.mean())
def draw(nd,k,rng,per=9):
    X=[];yy=[];gg=[];gid=0
    for c,n in enumerate(nd):
        for _ in range(n):
            b=rng.normal(0,np.sqrt(sb2))*k
            X.append(mu[c]+b+rng.normal(0,np.sqrt(sw2),(per,20))); yy+= [c]*per; gg+=[gid]*per; gid+=1
    return np.vstack(X),np.array(yy),np.array(gg)
def clf(): return SVC(C=10,gamma='scale')
out={}
for lab,nd in [('10/class',[10]*4),('real',[46,14,9,11]),('40/class',[40]*4)]:
    for k in [0,0.5,1,1.5]:
        bs=[];bd=[];tru=[]
        for rep in range(12):
            rng=np.random.default_rng(1000*rep+int(10*k))
            X,yy,gg=draw(nd,k,rng); Xn,yn,_=draw(nd,k,rng)
            t=np.mean(clf().fit(X,yy).predict(Xn)==yn)
            def cvacc(cv,groups=None):
                p=np.zeros(len(yy),int)
                for tr,te in (cv.split(X,yy) if groups is None else cv.split(X,yy,groups)): p[te]=clf().fit(X[tr],yy[tr]).predict(X[te])
                return np.mean(p==yy)
            bs.append(cvacc(StratifiedKFold(10,shuffle=True,random_state=rep))-t)
            bd.append(cvacc(StratifiedGroupKFold(10,shuffle=True,random_state=rep),gg)-t); tru.append(t)
        out[f'{lab}|{k}']=dict(bias_spec=float(np.mean(bs)),bias_spec_se=float(np.std(bs)/np.sqrt(12)),bias_don=float(np.mean(bd)),bias_don_se=float(np.std(bd)/np.sqrt(12)),true=float(np.mean(tru)))
        print(lab,k,{a:round(b*100,1) for a,b in out[f'{lab}|{k}'].items()},flush=True)
out['share']=share.tolist()
json.dump(out,open('simulate.json','w'),indent=1)
