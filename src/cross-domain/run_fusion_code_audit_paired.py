"""P1.12 companion to run_fusion_code_audit.py: Martian seed-44 paired TSS
differences between fusion configurations under three resampling units
(single cell-sol, whole cell, whole sol), B = 1000, rng seed 20260818.

    cd src/cross-domain
    python run_fusion_code_audit_paired.py

Writes outputs/fusion_code_audit_20261004T203942Z/P1_12_martian_paired_differences_by_block_unit.csv.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0,'.')
from uncertainty_analysis import block_indices, percentile_interval
S='outputs/uncertainty_sweep_20261004T092140396722Z/'
P=pd.read_csv('../martian/outputs/real/chapter6_supplementary_20261004T093226Z/predictions_test_and_my37_late.csv.gz')
q=P[(P.eval_set=='test')&(P.model=='lstm_w3')&(P.horizon==1)].reset_index(drop=True)
keys={'cell':q.cell_id.to_numpy(),'sol':(q.mars_year.astype(str)+'_'+q.sol.astype(str)).to_numpy()}
def tss(y,p,t):
    pr=p>=t; return pr[y==1].mean()-pr[y==0].mean()
def gidx(k,rng):
    u,inv=np.unique(k,return_inverse=True); m=[np.where(inv==i)[0] for i in range(len(u))]
    return np.concatenate([m[i] for i in rng.integers(0,len(u),len(u))])
d={c:pd.read_csv(S+f'predictions_{c}_martian_seed44.csv') for c in ['A1','A2','A3','A4','A5','A6','A7']}
rows=[]
for a,b in [('A7','A1'),('A7','A2'),('A7','A3'),('A7','A4'),('A6','A1'),('A5','A1'),('A2','A1')]:
    A,Bf=d[a],d[b]; y=A.y_true.to_numpy(); pa,pb=A.y_prob.to_numpy(),Bf.y_prob.to_numpy(); ta,tb=A.threshold[0],Bf.threshold[0]
    pt=tss(y,pa,ta)-tss(y,pb,tb); r={'comparison':f'{a}-{b}','diff':pt}
    for sch in ['one_cell_sol','cell','sol']:
        rng=np.random.default_rng(20260818); dist=[]
        for _ in range(1000):
            i=block_indices(len(y),1,rng) if sch=='one_cell_sol' else gidx(keys[sch],rng)
            dist.append(tss(y[i],pa[i],ta)-tss(y[i],pb[i],tb))
        lo,hi=percentile_interval(np.array(dist)); r[f'lo_{sch}']=lo; r[f'hi_{sch}']=hi
    rows.append(r)
R=pd.DataFrame(rows); R.to_csv('outputs/fusion_code_audit_20261004T203942Z/P1_12_martian_paired_differences_by_block_unit.csv',index=False); print(R.round(3).to_string(index=False))
