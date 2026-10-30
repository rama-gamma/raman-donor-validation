# New preprocessing: fingerprint (FP), high-wavenumber (HW), combined, and physically interpretable band features
import sys
import numpy as np
from scipy.signal import savgol_filter
from prep import baseline
d=np.load('raw.npz',allow_pickle=True)
shift,X=d['shift'][:1250],d['X'][:,:1250]   # drop duplicated last two columns
hist,donor,sample,point=d['hist'],d['donor'],d['sample'],d['point']
CL=['normal','dcis','ilc','idc']; y4=np.array([CL.index(h) for h in hist]); y2=(y4>0).astype(int)
def zs(A): return (A-A.mean(1,keepdims=True))/A.std(1,ddof=1,keepdims=True)
def prep_fp(order=5,sg=11,norm='z'):
    m=(shift>=800)&(shift<=2000); s=shift[m]; A=X[:,m]
    if sg: A=savgol_filter(A,sg,3,axis=1,mode='interp')
    A=np.array([baseline(s,a,order) for a in A])
    return s,normalise(s,A,norm)
def normalise(s,A,norm):
    if norm=='z': return zs(A)
    if norm=='l2': return A/np.linalg.norm(A,axis=1,keepdims=True)
    if norm=='area': return A/np.abs(A).sum(1,keepdims=True)*len(s)/100
    if norm=='1445':
        m=(s>=1425)&(s<=1465); return A/A[:,m].sum(1,keepdims=True)
def prep_hw(order=3):
    # baseline fitted on 2400-2990 cm-1 excluding the 2740-2805 detector artefact, kept 2810-2990
    m=(shift>=2400)&(shift<=2990)&~((shift>2740)&(shift<2805)); s=shift[m]
    A=savgol_filter(X[:,m],11,3,axis=1,mode='interp')
    A=np.array([baseline(s,a,order) for a in A])
    k=s>=2810; return s[k],A[:,k]
if __name__=='__main__':
    sF,F_raw=None,None
    m=(shift>=800)&(shift<=2000); sF=shift[m]
    A=savgol_filter(X[:,m],11,3,axis=1,mode='interp'); Fbc=np.array([baseline(sF,a,5) for a in A])
    sH,Hbc=prep_hw()
    np.savez('bc.npz',sF=sF,Fbc=Fbc,sH=sH,Hbc=Hbc)
    print(Fbc.shape,Hbc.shape)
