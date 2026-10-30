import numpy as np, warnings
from pcaopt import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
warnings.filterwarnings('ignore')
d=np.load('prep.npz',allow_pickle=True)
Y=d['Y']; hist=d['hist']; donor=d['donor']; sample=d['sample']; point=d['point']; s_axis=d['s']
CL=['normal','dcis','ilc','idc']
y4=np.array([CL.index(h) for h in hist]); y2=(y4>0).astype(int)
KMAX=100
def grid(name):
    K=[3,5,8,10,15,20,30,44,60,100]
    if name=='svm': return [dict(k=k,C=C,g=g) for k in K for C in [1,10,100] for g in [0.3,1,3]]
    if name=='lda': return [dict(k=k) for k in K]
    if name=='lr':  return [dict(k=k,C=C) for k in K for C in [0.01,0.1,1,10]]
    if name=='rf':  return [dict(k=k,mf=mf) for k in [5,10,20,44] for mf in ['sqrt',0.5]]
    if name=='pls': return [dict(k=k) for k in [2,3,5,8,10,15,20,30]]
class Fold:
    """PCA fitted once on training data; configs slice the first k scores (identical to PCA(k))."""
    def __init__(self,name,Xtr,ytr,Xte):
        self.name=name; self.Xtr,self.ytr,self.Xte=Xtr,ytr,Xte
        if name!='pls':
            p=PCA(min(KMAX,len(Xtr)-1)).fit(Xtr); self.Ztr=p.transform(Xtr); self.Zte=p.transform(Xte)
    def model(self,p):
        n=self.name
        if n=='svm':
            Z=self.Ztr[:,:p['k']]; g=p['g']/(Z.shape[1]*Z.var()); return SVC(C=p['C'],gamma=g,probability=False)
        if n=='lda': return LDA(solver='svd')
        if n=='lr': return LogisticRegression(C=p['C'],max_iter=3000)
        if n=='rf': return RandomForestClassifier(200,max_features=p['mf'],random_state=0,n_jobs=1)
    def fit(self,p):
        if self.name=='pls':
            cl=np.unique(self.ytr); T=(self.ytr[:,None]==cl[None]).astype(float)
            m=PLSRegression(p['k'],scale=False).fit(self.Xtr,T)
            self.dec=lambda X: m.predict(X); self.cl=cl; return self
        m=self.model(p).fit(self.Ztr[:,:p['k']],self.ytr); self.m=m; self.k=p['k']; return self
    def predict(self,p):
        self.fit(p)
        if self.name=='pls': return self.cl[self.dec(self.Xte).argmax(1)]
        return self.m.predict(self.Zte[:,:self.k])
def fit_pred(name,p,Xtr,ytr,Xte): return Fold(name,Xtr,ytr,Xte).predict(p)
def select(name,Xtr,ytr,gtr,seed,nfold=4):
    G=grid(name); sc=np.zeros(len(G))
    for itr,ite in StratifiedGroupKFold(nfold,shuffle=True,random_state=100+seed).split(Xtr,ytr,gtr):
        F=Fold(name,Xtr[itr],ytr[itr],Xtr[ite])
        for j,p in enumerate(G): sc[j]+=np.mean(F.predict(p)==ytr[ite])
    return G[int(sc.argmax())]
