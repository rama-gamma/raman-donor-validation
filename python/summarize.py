import json,glob,numpy as np
from common import *
S={}
rng=np.random.default_rng(0)
for f in sorted(glob.glob('nested_*.json')):
    name,task=f[7:-5].split('_'); y=y4 if task=='four' else y2; nc=len(np.unique(y))
    R=json.load(open(f)); P=np.array([r['pred'] for r in R])
    acc=(P==y).mean(1); bal=np.array([np.mean([np.mean(p[y==c]==c) for c in range(nc)]) for p in P])
    sens=np.array([[np.mean(p[y==c]==c) for c in range(nc)] for p in P]).mean(0)
    # sample-level majority vote
    us=np.unique(sample); ys=np.array([y[sample==u][0] for u in us])
    sa=[];ssens=[]
    for p in P:
        v=np.array([np.bincount(p[sample==u],minlength=nc).argmax() for u in us]); sa.append(np.mean(v==ys)); ssens.append([np.mean(v[ys==c]==c) for c in range(nc)])
    # donor bootstrap CI on pooled predictions (accuracy)
    dn=np.unique(donor); bs=[]
    corr=(P==y).mean(0)
    for b in range(2000):
        dd=rng.choice(dn,len(dn)); idx=np.concatenate([np.where(donor==d_)[0] for d_ in dd]); bs.append(corr[idx].mean())
    cm=np.array([[np.mean(P[:,y==t]==pp) for pp in range(nc)] for t in range(nc)])
    ks=[c['k'] for r in R for c in r['chosen']]
    S[f'{name}_{task}']=dict(n_seeds=len(R),acc=acc.mean(),acc_sd=acc.std(),bal=bal.mean(),sens=sens.tolist(),
        ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],sample_acc=float(np.mean(sa)),sample_sens=np.mean(ssens,0).tolist(),
        n_samples=len(us),cm=cm.tolist(),k_median=float(np.median(ks)),chosen=R[0]['chosen'][:3])
    print(f"{name:4s} {task:6s} seeds={len(R)} acc={acc.mean()*100:.1f}±{acc.std()*100:.1f} bal={bal.mean()*100:.1f} CI={S[f'{name}_{task}']['ci'][0]*100:.1f}-{S[f'{name}_{task}']['ci'][1]*100:.1f} sampleacc={np.mean(sa)*100:.1f} sens={np.round(sens*100,1)} ssens={np.round(np.mean(ssens,0)*100,1)} kmed={np.median(ks)}")
json.dump(S,open('summary.json','w'),indent=1)
print('majority four', np.mean(y4==0), 'binary', np.mean(y2==1))
