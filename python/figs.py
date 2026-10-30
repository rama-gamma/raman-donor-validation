import json,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import *
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
FD='../figures/'; O='../octave/results/'
BL='#2E5A87'; RD='#C8553D'; GR='#9A9A9A'; LB='#8FB3D9'
S=json.load(open('summary.json')); SP=json.load(open('split_levels.json'))
def seeds(f):
    a=np.loadtxt(O+f,delimiter=',',ndmin=2); return 100*a[:,1].mean(),100*a[:,1].std(),a[:,2].mean()
pub={'svm_binary':97.7,'svm_four':91.0,'lda_binary':88.2,'lda_four':79.4}
# ---------- Fig A: bug + accuracy ----------
def figA(path,ma=False):
    fig=plt.figure(figsize=(7.2,3.0)); ax=fig.add_axes([0.02,0.12,0.40,0.78]); ax.axis('off')
    ids=['S1','S1','S2','S2','S3','S3','S4','S4','S5','S5']; test=[1,0,0,0,0,0,1,0,0,0]
    used=[1,2,3,4,6,7,8]; intended=[3,4,5,6,9,10]
    rows=[('Fold assignment',[('T' if t else '') for t in test],[RD if t else '#E8E8E8' for t in test]),
          ('Intended training rows',['' for _ in ids],[BL if (i+1) in intended else 'white' for i in range(10)]),
          ('Rows used by deposited code',['' for _ in ids],[(RD if test[i] else BL) if (i+1) in used else 'white' for i in range(10)])]
    for r,(lab,txt,col) in enumerate(rows):
        yy=2-r; ax.text(0,yy+0.62,lab,fontsize=8,fontweight='bold')
        for i in range(10):
            ax.add_patch(Rectangle((i*0.95,yy),0.85,0.45,fc=col[i],ec='#555',lw=0.6))
            ax.text(i*0.95+0.42,yy+0.22,txt[i] if txt[i] else ids[i],ha='center',va='center',fontsize=6.5,color='white' if col[i] in (RD,BL) else '#333')
    ax.text(0,-0.45,'Toy example run in GNU Octave 11.1: trainIdx(removeIdx) = [] deletes\nmask elements, so test rows 1 and 7 (red) enter the training set.',fontsize=7,color='#333')
    ax.set_xlim(-0.1,9.6); ax.set_ylim(-0.7,3.0)
    ax.text(-0.1,3.0,'a',fontsize=11,fontweight='bold')
    b=fig.add_axes([0.53,0.17,0.45,0.70])
    keys=[('svm_binary','SVM\ncancer vs normal'),('svm_four','SVM\nfour-class'),('lda_four','LDA\nfour-class')]
    import os
    if not ma and os.path.exists(O+'/seeds_lda_binary_deposited.csv'): keys.insert(2,('lda_binary','LDA\ncancer vs normal'))
    w=0.2; x=np.arange(len(keys))
    for j,(k,_) in enumerate(keys):
        dep=seeds(f'seeds_{k}_deposited.csv'); fx=seeds(f'seeds_{k}_fixed.csv'); ne=S[k]['acc']*100
        vals=[pub[k],dep[0],fx[0],ne]; errs=[0,dep[1],fx[1],S[k]['acc_sd']*100]
        for i,(v,e,c) in enumerate(zip(vals,errs,[GR,RD,LB,BL])):
            b.bar(x[j]+(i-1.5)*w,v,w,color=c,yerr=e if e else None,capsize=1.5,error_kw=dict(lw=0.6),label=['Published','Deposited code re-run (Octave)','Indexing line corrected','Donor-level, nested'][i] if j==0 else None)
            b.text(x[j]+(i-1.5)*w,v+1.2,f'{v:.0f}',ha='center',fontsize=6)
    b.set_xticks(x); b.set_xticklabels([l for _,l in keys],fontsize=7.5); b.set_ylim(0,112); b.set_yticks(range(0,101,20)); b.set_ylabel('Accuracy (%)')
    b.legend(fontsize=6.5,frameon=False,loc='upper center',ncol=2,bbox_to_anchor=(0.5,1.16))
    b.text(-0.15,1.13,'b',transform=b.transAxes,fontsize=11,fontweight='bold')
    fig.savefig(path,bbox_inches='tight'); plt.close(fig)
# ---------- Fig B: per-class sensitivity, nested donor-level, 5 classifiers ----------
def figB(path):
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.7),gridspec_kw=dict(width_ratios=[1.6,1]))
    ms=[('svm','SVM'),('lr','Logistic'),('lda','LDA'),('pls','PLS-DA'),('rf','Random forest')]
    cols=['#2E5A87','#5B86B0','#8FB3D9','#C8553D','#E3A08F']
    ax=axs[0]; x=np.arange(4); w=0.16
    for i,(m,l) in enumerate(ms):
        v=np.array(S[f'{m}_four']['sens'])*100; ax.bar(x+(i-2)*w,v,w,color=cols[i],label=l)
    ax.set_xticks(x); ax.set_xticklabels(['Normal','DCIS','ILC','IDC']); ax.set_ylabel('Sensitivity on unseen donors (%)'); ax.set_ylim(0,100)
    ax.legend(fontsize=6.5,frameon=False,ncol=5,loc='upper center',bbox_to_anchor=(0.5,1.14))
    ax.text(-0.12,1.08,'a',transform=ax.transAxes,fontsize=11,fontweight='bold')
    ax=axs[1]; cm=np.array(S['svm_four']['cm'])*100
    ax.imshow(cm,cmap='Blues',vmin=0,vmax=100)
    for i in range(4):
        for j in range(4): ax.text(j,i,f'{cm[i,j]:.0f}',ha='center',va='center',fontsize=7.5,color='white' if cm[i,j]>55 else '#222')
    t=['Normal','DCIS','ILC','IDC']; ax.set_xticks(range(4)); ax.set_xticklabels(t,fontsize=7); ax.set_yticks(range(4)); ax.set_yticklabels(t,fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('SVM, row %',fontsize=8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.text(-0.35,1.08,'b',transform=ax.transAxes,fontsize=11,fontweight='bold')
    fig.tight_layout(); fig.savefig(path,bbox_inches='tight'); plt.close(fig)
# ---------- Fig C: splitting unit ladder ----------
def figC(path):
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.5))
    units=['spectrum','point','sample','donor']; lab=['Spectrum','Measurement\npoint','Sample','Donor']
    for ax,task,tt in [(axs[0],'binary','Cancer vs normal'),(axs[1],'four','Four-class')]:
        v=[SP[f'{task}_{u}']['mean']*100 for u in units]; e=[SP[f'{task}_{u}']['sd']*100 for u in units]
        ax.bar(range(4),v,0.6,yerr=e,color=[RD,'#E3A08F',LB,BL],capsize=2)
        for i,a in enumerate(v): ax.text(i,a+1.5,f'{a:.1f}',ha='center',fontsize=7)
        ax.set_xticks(range(4)); ax.set_xticklabels(lab,fontsize=7); ax.set_ylim(0,105); ax.set_title(tt,fontsize=8.5); ax.set_ylabel('Accuracy (%)')
    axs[0].text(-0.15,1.05,'a',transform=axs[0].transAxes,fontsize=11,fontweight='bold'); axs[1].text(-0.15,1.05,'b',transform=axs[1].transAxes,fontsize=11,fontweight='bold')
    fig.tight_layout(); fig.savefig(path,bbox_inches='tight'); plt.close(fig)
# ---------- Fig D: PC sweeps ----------
def figD(path):
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.5))
    for ax,(k,tt) in zip(axs,[('svm_four','SVM, four-class'),('lda_four','LDA, four-class')]):
        for md,c,l in [('deposited',RD,'Deposited code'),('fixed',LB,'Indexing line corrected')]:
            a=np.loadtxt(O+f'sweep_{k}_{md}.csv',delimiter=','); ax.plot(a[:,0],a[:,1]*100,color=c,lw=1.2,label=l)
        ax.axhline(pub[k],color=GR,ls='--',lw=0.8); ax.text(300,pub[k]+1,'published',ha='right',fontsize=6.5,color='#555')
        ax.axhline(S[k]['acc']*100,color=BL,ls=':',lw=1); ax.text(300,S[k]['acc']*100-5,'donor-level nested',ha='right',fontsize=6.5,color=BL)
        ax.set_xlabel('Number of principal components'); ax.set_ylabel('Accuracy (%)'); ax.set_title(tt,fontsize=8.5); ax.set_ylim(30,100)
    axs[0].legend(fontsize=6.5,frameon=False,loc='lower right',bbox_to_anchor=(1,0.12))
    fig.tight_layout(); fig.savefig(path,bbox_inches='tight'); plt.close(fig)
# ---------- Fig E: conformal ----------
def figE(path):
    C=json.load(open('conformal.json'))['four']
    fig,ax=plt.subplots(figsize=(4.6,2.6)); x=np.arange(4); w=0.26
    for i,(k,l,c) in enumerate([('cov_class','Spectrum, marginal',LB),('mcov_class','Spectrum, class-conditional',BL),('s_cov_class','Sample, marginal',RD)]):
        v=np.array(C[k])*100; ax.bar(x+(i-1)*w,v,w,color=c,label=l)
    ax.axhline(90,color='#333',ls='--',lw=0.8); ax.set_xticks(x); ax.set_xticklabels(['Normal','DCIS','ILC','IDC']); ax.set_ylim(0,105); ax.set_ylabel('Coverage (%)')
    ax.set_ylim(0,105); ax.legend(fontsize=6.5,frameon=False,loc='upper center',ncol=3,bbox_to_anchor=(0.55,1.13)); fig.tight_layout(); fig.savefig(path,bbox_inches='tight'); plt.close(fig)
# ---------- Fig F: simulation ----------
def figF(path):
    M=json.load(open('simulate.json')); ks=[0,0.5,1,1.5]
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.5),sharey=True)
    for ax,key,tt in [(axs[0],'bias_spec','Spectrum-level CV'),(axs[1],'bias_don','Donor-level CV')]:
        for lab,c,mk in [('10/class',BL,'o'),('real',RD,'s'),('40/class','#5A9A5A','^')]:
            v=[M[f'{lab}|{k}'][key]*100 for k in ks]; e=[M[f'{lab}|{k}'][key+'_se']*100 for k in ks]
            ax.errorbar(ks,v,yerr=e,color=c,marker=mk,ms=3.5,lw=1,capsize=2,label={'10/class':'10 donors per class','real':'Real class sizes (46/14/9/11)','40/class':'40 donors per class'}[lab])
        ax.axhline(0,color='#333',lw=0.6); ax.axvline(1,color=GR,ls=':',lw=0.8); ax.set_title(tt,fontsize=8.5); ax.set_xlabel('Donor-effect scale (1 = estimated)')
    axs[0].set_ylabel('CV accuracy − new-donor accuracy (points)'); axs[0].legend(fontsize=6.5,frameon=False)
    fig.tight_layout(); fig.savefig(path,bbox_inches='tight'); plt.close(fig)
# ---------- Fig G: spectra ----------
def figG(path):
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.5))
    cols=['#2E5A87','#8FB3D9','#C8553D','#E3A08F']
    for c in range(4):
        axs[0].plot(s_axis,Y[y4==c].mean(0)+3*(3-c),color=cols[c],lw=1); axs[0].text(2005,3*(3-c),CL[c].upper() if c else 'Normal',fontsize=7,color=cols[c],va='bottom')
    axs[0].set_yticks([]); axs[0].set_xlabel('Raman shift (cm$^{-1}$)'); axs[0].set_title('Class means after preprocessing (offset)',fontsize=8.5); axs[0].set_xlim(800,2120)
    Z=PCA(20).fit_transform(Y)
    axs[1].scatter(Z[:,0],Z[:,1],s=4,color='#DDDDDD')
    ids=[d_ for d_ in np.unique(donor[y4==3])][:6]; pal=plt.cm.tab10.colors
    for i,d_ in enumerate(ids): m=donor==d_; axs[1].scatter(Z[m,0],Z[m,1],s=7,color=pal[i])
    axs[1].set_xlabel('PC1'); axs[1].set_ylabel('PC2'); axs[1].set_title('Six IDC donors highlighted',fontsize=8.5)
    axs[0].text(-0.05,1.05,'a',transform=axs[0].transAxes,fontsize=11,fontweight='bold'); axs[1].text(-0.15,1.05,'b',transform=axs[1].transAxes,fontsize=11,fontweight='bold')
    fig.tight_layout(); fig.savefig(path,bbox_inches='tight'); plt.close(fig)
import sys
which=sys.argv[1:] or ['A','B','C','D','E','F','G']
for w in which:
    if w=='A': figA(FD+'Fig2.png')
    if w=='B': figB(FD+'Fig4.png')
    if w=='C': figC(FD+'Fig3.png')
    if w=='D': figD(FD+'FigS1.png')
    if w=='E': figE(FD+'Fig5.png')
    if w=='F': figF(FD+'Fig6.png')
    if w=='G': figG(FD+'Fig1.png')
    print('ok',w)
