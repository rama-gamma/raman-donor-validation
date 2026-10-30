import numpy as np, pandas as pd
import os
DATA=os.environ.get('RAMAN_DATA',os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data','DATASET_ID UPDATED v2.xlsx'))
from scipy.signal import savgol_filter
def load():
    df=pd.read_excel(DATA,sheet_name=0,header=None)
    shift=df.iloc[0,2:1252].astype(float).values
    X=df.iloc[1:,2:1252].astype(float).values
    hist=df.iloc[1:,0].str.lower().values
    sid=df.iloc[1:,1].astype(str).values
    donor=df.iloc[1:,1254].astype(int).values
    sample=np.array([s[:4] for s in sid]); point=np.array([s[:7] for s in sid])
    return shift,X,hist,sid,donor,sample,point
def baseline(x,y,order=5):
    def fit(xx,yy):
        xm,xs=xx.mean(),xx.std(ddof=1)
        p=np.polyfit((xx-xm)/xs,yy,order); return lambda q: np.polyval(p,(q-xm)/xs)
    f=fit(x,y); b0=f(x); r=y-b0; dev0=np.sqrt(np.mean((r-r.mean())**2))
    m=y<=b0; xr=x[m]; yr=y[m].copy(); devs=[]; j=0
    while True:
        f=fit(xr,yr); yf=f(xr); r=yr-yf; d=np.sqrt(np.mean((r-r.mean())**2)); devs.append(d)
        cont = abs(d-dev0)/d>0.05 if j==0 else abs((d-devs[-2])/d)>0.05
        idx=yr>=yf; yr[idx]=yf[idx]; j+=1
        if not cont or j>500: break
    return y-f(x)
def preprocess(shift,X):
    m=(shift>=800)&(shift<=2000); s=shift[m]; Y=X[:,m]
    Y=savgol_filter(Y,11,3,axis=1,mode='interp')
    Y=np.array([baseline(s,y) for y in Y])
    Y=(Y-Y.mean(1,keepdims=True))/Y.std(1,ddof=1,keepdims=True)
    return s,Y
if __name__=='__main__':
    shift,X,hist,sid,donor,sample,point=load()
    s,Y=preprocess(shift,X)
    np.savez('prep.npz',s=s,Y=Y,hist=hist,sid=sid,donor=donor,sample=sample,point=point)
    np.savez('raw.npz',shift=shift,X=X,hist=hist,sid=sid,donor=donor,sample=sample,point=point)
    # inputs for the GNU Octave re-execution (octave/): raw spectra, numeric labels, sample identifiers
    od=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','octave')
    np.savetxt(os.path.join(od,'raw.csv'),np.vstack([shift,X]),delimiter=',',fmt='%.10g')
    lab={'normal':1,'dcis':2,'ilc':3,'idc':4}
    np.savetxt(os.path.join(od,'label.csv'),np.array([lab[h] for h in hist]),fmt='%d')
    open(os.path.join(od,'sid.txt'),'w').write('\n'.join(sample)+'\n')   # = extractBetween(sample_id,1,4) in main_code.mlx
    import collections
    print(Y.shape, collections.Counter(hist))
    for h in ['normal','dcis','ilc','idc']: print(h,len(set(donor[hist==h])),len(set(sample[hist==h])),len(set(point[hist==h])))
    print('donors',len(set(donor)),'samples',len(set(sample)),'points',len(set(point)))
    # samples->donor unique?
    print(pd.Series(donor).groupby(sample).nunique().max())
