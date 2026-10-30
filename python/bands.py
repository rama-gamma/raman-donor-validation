import numpy as np, json
from feats import *
from scipy.stats import mannwhitneyu
b=np.load('bc.npz'); sF,Fbc,sH,Hbc=b['sF'],b['Fbc'],b['sH'],b['Hbc']
def area(s,A,lo,hi):
    m=(s>=lo)&(s<=hi); return np.clip(A[:,m].mean(1),1.0,None)
F=(sF,Fbc); H=(sH,Hbc)
BANDS={'1004':(F,998,1010),'1157':(F,1150,1165),'1265':(F,1255,1275),'1302':(F,1295,1310),'1445':(F,1435,1455),
 '1525':(F,1515,1530),'1655':(F,1645,1665),'1745':(F,1738,1752),'2850':(H,2845,2858),'2885':(H,2878,2892),'2935':(H,2925,2945)}
Aband={k:area(*v[0],v[1],v[2]) for k,v in BANDS.items()}
RAT={'I1655/I1445':('1655','1445'),'I1745/I1445':('1745','1445'),'I1004/I1445':('1004','1445'),
 'I1265/I1302':('1265','1302'),'I1525/I1445':('1525','1445'),'I2850/I2935':('2850','2935'),'I2885/I2850':('2885','2850')}
R=np.column_stack([np.log(Aband[a]/Aband[b_]) for a,b_ in RAT.values()])
np.save('ratios.npy',R)
names=list(RAT)
# donor-level class means (one value per donor-class unit) and DCIS vs ILC test
unit=np.array([f'{d_}_{h}' for d_,h in zip(donor,hist)])
U=np.unique(unit); Ru=np.array([R[unit==u].mean(0) for u in U]); hu=np.array([u.split('_')[1] for u in U])
out={'names':names,'class_mean':{},'dcis_vs_ilc':[],'normal_vs_cancer':[],'ilc_vs_idc':[]}
for h in CL: out['class_mean'][h]=[float(np.mean(np.exp(Ru[hu==h][:,j]))) for j in range(len(names))]
def comp(a,b_,key):
    for j,nm in enumerate(names):
        x,y=Ru[hu==a][:,j],Ru[np.isin(hu,b_)][:,j]; p=mannwhitneyu(x,y).pvalue
        # rank-biserial effect size
        u=mannwhitneyu(x,y).statistic; rb=2*u/(len(x)*len(y))-1
        out[key].append(dict(name=nm,p=float(p),rb=float(rb),mean_a=float(np.exp(x.mean())),mean_b=float(np.exp(y.mean()))))
comp('dcis',['ilc'],'dcis_vs_ilc'); comp('normal',['dcis','ilc','idc'],'normal_vs_cancer'); comp('ilc',['idc'],'ilc_vs_idc')
# donor variance share per wavenumber within class (one-way random effects), and class share at donor level
def icc(Z):
    res=np.zeros(Z.shape[1]); cls=np.zeros(Z.shape[1])
    mu=np.array([Z[y4==c].mean(0) for c in range(4)]); Rr=Z-mu[y4]
    dm={u:Rr[unit==u].mean(0) for u in U}; W=np.array([Rr[i]-dm[unit[i]] for i in range(len(Rr))])
    sw2=W.var(0,ddof=1); nbar=len(Z)/len(U); B=np.array(list(dm.values())); sb2=np.maximum(B.var(0,ddof=1)-sw2/nbar,0)
    Du=np.array([Z[unit==u].mean(0) for u in U]); grand=Du.mean(0)
    mc=np.array([Du[hu==h].mean(0) for h in CL]); nc=np.array([np.sum(hu==h) for h in CL])
    sbc=(nc[:,None]*(mc-grand)**2).sum(0)/len(U)
    return sb2/(sb2+sw2), sbc/Du.var(0)
FPz=zs(Fbc); HWz=zs(Hbc)
iF,cF=icc(FPz); iH,cH=icc(HWz)
out['icc_fp']=iF.tolist(); out['cls_fp']=cF.tolist(); out['icc_hw']=iH.tolist(); out['cls_hw']=cH.tolist()
out['sF']=sF.tolist(); out['sH']=sH.tolist()
out['icc_fp_mean']=float(iF.mean()); out['icc_hw_mean']=float(iH.mean())
json.dump(out,open('bands.json','w'),indent=1)
print('ICC fp mean',iF.mean().round(3),'hw',iH.mean().round(3))
for key in ['dcis_vs_ilc','ilc_vs_idc','normal_vs_cancer']:
    print(key)
    for r in out[key]: print(' ',r['name'].ljust(14),round(r['mean_a'],3),round(r['mean_b'],3),'p=%.3g rb=%.2f'%(r['p'],r['rb']))
for h in CL: print(h,np.round(out['class_mean'][h],3))
# top donor-dominated and class-dominated wavenumbers
for nm,s_,i_,c_ in [('FP',sF,iF,cF),('HW',sH,iH,cH)]:
    print(nm,'max ICC at',s_[np.argsort(i_)[-5:]],'max class share at',s_[np.argsort(c_)[-5:]])
