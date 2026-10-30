import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from feats import *
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
BL='#2E5A87'; RD='#C8553D'; GR='#9A9A9A'; LB='#8FB3D9'; GN='#5A9A5A'
CC={'normal':GR,'dcis':BL,'ilc':RD,'idc':GN}; LAB={'normal':'Normal','dcis':'DCIS','ilc':'ILC','idc':'IDC'}
b=np.load('bc.npz'); sF,Fbc,sH,Hbc=b['sF'],b['Fbc'],b['sH'],b['Hbc']; B=json.load(open('bands.json'))
R=np.load('ratios.npy'); names=B['names']
# ---------- Fig 7 ----------
fig=plt.figure(figsize=(7.2,7.9))
a1=fig.add_axes([0.08,0.72,0.62,0.245]); a2=fig.add_axes([0.74,0.72,0.24,0.245],sharey=a1)
for h in CL:
    m=hist==h; a1.plot(sF,Fbc[m].mean(0),color=CC[h],lw=1,label=LAB[h]); a2.plot(sH,Hbc[m].mean(0)/3,color=CC[h],lw=1)
for w,t in [(1004,'1004'),(1157,'1157'),(1265,'1265'),(1302,'1302'),(1445,'1445'),(1525,'1525'),(1655,'1655'),(1745,'1745')]:
    a1.axvline(w,color='#bbb',lw=.5,zorder=0); a1.text(w,2080,t,rotation=90,fontsize=6,ha='center',va='bottom',color='#555')
for w in [2850,2885,2935]:
    a2.axvline(w,color='#bbb',lw=.5,zorder=0); a2.text(w,2080,str(w),rotation=90,fontsize=6,ha='center',va='bottom',color='#555')
a1.set_ylim(-50,2550); a1.set_xlim(800,2000); a2.set_xlim(2810,2990)
a1.set_ylabel('Baseline-corrected counts'); a1.set_xlabel('Raman shift (cm$^{-1}$)'); a2.set_xlabel('Raman shift (cm$^{-1}$)')
a2.text(2815,2420,'×1/3',fontsize=7,color='#555'); plt.setp(a2.get_yticklabels(),visible=False)
a1.legend(frameon=False,fontsize=7,loc='center left',bbox_to_anchor=(0.0,0.62)); a1.text(-0.07,1.08,'a',transform=a1.transAxes,fontsize=11,fontweight='bold')
b1=fig.add_axes([0.08,0.43,0.62,0.21]); b2=fig.add_axes([0.74,0.43,0.24,0.21],sharey=b1)
b1.plot(sF,B['icc_fp'],color=RD,lw=1,label='Donor share (within class)'); b1.plot(sF,B['cls_fp'],color=BL,lw=1,label='Class share (donor means)')
b2.plot(sH,B['icc_hw'],color=RD,lw=1); b2.plot(sH,B['cls_hw'],color=BL,lw=1)
b1.set_xlim(800,2000); b2.set_xlim(2810,2990); b1.set_ylim(0,1); b1.set_ylabel('Variance share'); b1.set_xlabel('Raman shift (cm$^{-1}$)'); b2.set_xlabel('Raman shift (cm$^{-1}$)')
plt.setp(b2.get_yticklabels(),visible=False); b1.legend(frameon=False,fontsize=7,loc='upper left',ncol=2)
for w in [1004,1157,1525]: b1.axvline(w,color='#bbb',lw=.5,zorder=0)
b1.text(-0.07,1.08,'b',transform=b1.transAxes,fontsize=11,fontweight='bold')
unit=np.array([f'{d_}_{h}' for d_,h in zip(donor,hist)]); U=np.unique(unit); hu=np.array([u.split('_')[1] for u in U])
Ru=np.array([R[unit==u].mean(0) for u in U])
sel=[2,0,4,5]; ttl=['I$_{1004}$/I$_{1445}$\n(Phe / CH$_2$)','I$_{1655}$/I$_{1445}$\n(amide I, C=C / CH$_2$)','I$_{1525}$/I$_{1445}$\n(carotenoid / CH$_2$)','I$_{2850}$/I$_{2935}$\n(CH$_2$ / CH$_3$ stretch)']
rng=np.random.default_rng(1)
for k,j in enumerate(sel):
    ax=fig.add_axes([0.08+k*0.235,0.055,0.19,0.235])
    for i,h in enumerate(CL):
        v=np.exp(Ru[hu==h][:,j]); ax.boxplot(v,positions=[i],widths=.55,showfliers=False,medianprops=dict(color='k',lw=1),boxprops=dict(lw=.7),whiskerprops=dict(lw=.7),capprops=dict(lw=.7))
        ax.scatter(i+rng.uniform(-.18,.18,len(v)),v,s=7,color=CC[h],alpha=.8,lw=0,zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels([LAB[h] for h in CL],fontsize=6.5,rotation=0); ax.set_title(ttl[k],fontsize=7.5)
    if j in (2,4): ax.set_yscale('log')
    if k==0: ax.set_ylabel('Donor-mean ratio'); ax.text(-0.35,1.18,'c',transform=ax.transAxes,fontsize=11,fontweight='bold')
plt.savefig('../figures/Fig7.png'); plt.close()
# ---------- Fig 8 ----------
S=json.load(open('regions_summary.json')); L=json.load(open('lcurve.json')); P=json.load(open('sim_phys.json'))
fig,ax=plt.subplots(1,3,figsize=(7.4,3.1),gridspec_kw=dict(width_ratios=[1.25,1,1.15]))
fs=['FP','HW','FP+HW','ratios']; fl=['FP','HW','FP+HW','7 ratios']; x=np.arange(4); w=0.2
for i,(key,task,met,col,lab) in enumerate([('lr','binary','acc',LB,'Cancer vs normal, accuracy'),('lr','four','bal',BL,'Four-class, balanced acc.'),('lr','four','ilc',RD,'ILC sensitivity')]):
    v=[100*(S[f'{key}|{f}|{task}']['sens'][2] if met=='ilc' else S[f'{key}|{f}|{task}'][met]) for f in fs]
    ax[0].bar(x+(i-1)*w,v,w,color=col,label=lab)
    for xx,vv in zip(x+(i-1)*w,v): ax[0].text(xx,vv+1,f'{vv:.0f}',ha='center',fontsize=5.8)
ax[0].set_xticks(x); ax[0].set_xticklabels(fl,fontsize=7); ax[0].set_ylim(0,105); ax[0].set_ylabel('%'); ax[0].legend(frameon=False,fontsize=6,loc='upper right',bbox_to_anchor=(1.0,1.22),ncol=1)
ax[0].text(-0.2,1.08,'a',transform=ax[0].transAxes,fontsize=11,fontweight='bold')
fr=[0.25,0.5,0.75,1.0]; nI=[L[f'four|{f}']['ntr'][2] for f in fr]
for c,h in enumerate(CL): ax[1].plot([100*f for f in fr],[100*L[f'four|{f}']['sens'][c] for f in fr],'o-',ms=3,color=CC[h],label=LAB[h])
ax[1].plot([100*f for f in fr],[100*L[f'binary|{f}']['acc'] for f in fr],'s--',ms=3,color='k',lw=.8,label='Cancer vs normal')
ax[1].set_xlabel('Training donors used (%)'); ax[1].set_ylabel('Sensitivity / accuracy (%)'); ax[1].set_ylim(0,100); ax[1].set_xticks([25,50,75,100])
for f,n in zip(fr,nI): ax[1].text(100*f,3,f'{n:.0f}',ha='center',fontsize=6,color=RD)
ax[1].text(14,9,'ILC donors:',fontsize=6,color=RD); ax[1].legend(frameon=False,fontsize=6,loc='center right',bbox_to_anchor=(1.02,0.42))
ax[1].text(-0.2,1.08,'b',transform=ax[1].transAxes,fontsize=11,fontweight='bold')
keys=['shot noise only|q=1.0','donor background|q=1.0','donor gain|q=1.0','donor chemistry|q=1.0','all donor effects|q=1.0','all donor effects|q=0.1','all donor effects|q=10.0']
kl=['Shot noise\nonly','Donor\nbackground','Donor\ngain','Donor\nchemistry','All','All,\n0.1× photons','All,\n10× photons']
y=np.arange(len(keys))[::-1]
ax[2].barh(y+0.18,[100*P[k]['bias_spec'] for k in keys],0.36,xerr=[100*P[k]['se_spec'] for k in keys],color=RD,label='Spectrum-level CV',error_kw=dict(lw=.6))
ax[2].barh(y-0.18,[100*P[k]['bias_don'] for k in keys],0.36,xerr=[100*P[k]['se_don'] for k in keys],color=BL,label='Donor-level CV',error_kw=dict(lw=.6))
ax[2].set_yticks(y); ax[2].set_yticklabels(kl,fontsize=6.3); ax[2].axvline(0,color='k',lw=.6); ax[2].set_xlabel('CV − new-donor acc. (points)')
ax[2].legend(frameon=False,fontsize=6,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=2); ax[2].text(-0.42,1.08,'c',transform=ax[2].transAxes,fontsize=11,fontweight='bold')
plt.tight_layout(); plt.savefig('../figures/Fig8.png'); plt.close()
