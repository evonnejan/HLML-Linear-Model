"""No training, no checkpoint access, only data loading and deterministic probes.
run.py is parsed as text only; its entry point is never imported or executed.
"""
import os, sys, ast, json, fnmatch, argparse, contextlib, io
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path.cwd()))
from data_provider.Data_Loader import Dataset_Custom
from data_provider.Data_Factory import data_provider
from build_splits import count_windows_per_segment,assign_split,assign_folds,blocked_boundaries,find_overlap_groups,pick_boundary
from rebuild_gate_columns import merge_gate_per_column
from models.DLinearMix2 import Model
from utils.metrics import CORR

OUT=Path('docs/review/reports/2026-09-13-r01/evidence')
def emit(k,v):print(k,json.dumps(v,ensure_ascii=False,default=str))
def quiet(fn,*a,**kw):
    with contextlib.redirect_stdout(io.StringIO()):return fn(*a,**kw)
def extract(path,names,cls=None,env=None):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    body=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body if cls else tree.body
    nodes=[n for n in body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    for n in nodes:n.decorator_list=[]
    ns=dict(globals());ns.update(env or {})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
    return ns

core=extract('run.py',['_parse_csv_cols','_expand_col_patterns','_configure_mix_model_args','_drop_constant_columns','_validate_split_args'])
headers=list(pd.read_csv('dataset/train_drycut_L3h_buf60.csv',nrows=0).columns)
args=SimpleNamespace(input_col=','.join(core['_expand_col_patterns'](['HL*'],headers)),exog_col=None,model='DLinearMix2',branch_in=None,exog_in=None,mix_in=None,target='HL01')
core['_configure_mix_model_args'](args)
emit('target_guard',{'input':args.input_col,'target':args.target,'accepted':True})
gates=[c for c in headers if 'gate_opening' in c]
exog=['min_since_rain','Past10Min','Past1Hr','Now']+gates
common=dict(root_path='dataset',size=[96,30,15],features='S',input_col='HL02,HL03,HL04,HL05,HL06',exog_col=','.join(exog),segment_col='segment_id',target='HL01',model_name='DLinearMix2',freq='min')
for tag in ['drycut_L3h_buf60','old']:
    train=pd.read_csv(f'dataset/train_{tag}.csv')
    sp=pd.read_csv(f'dataset/splits_train_{tag}.csv').fillna('')
    counts=count_windows_per_segment(train,seg_col='segment_id',check_cols=['HL01','HL02','HL03','HL04','HL05','HL06']+exog,need=111,stride=1)
    win=train[['segment_id','WinStart','WinEnd']].drop_duplicates().copy()
    for c in ['WinStart','WinEnd']:win[c]=pd.to_datetime(win[c])
    blocked=blocked_boundaries(counts,find_overlap_groups(win,'segment_id'),'segment_id')
    regen=assign_folds(assign_split(counts,(.7,.15,.15),blocked),3,.5,blocked)
    encoded=regen.to_csv(index=False).encode()
    emit('split_regenerate',{'tag':tag,'byte_equal':encoded==Path(f'dataset/splits_train_{tag}.csv').read_bytes()})
    legacy_ids=sp.segment_id.tolist()[:int(len(sp)*.7)]
    for fold in [None,1,2,3]:
        col='split' if fold is None else f'fold_{fold}'
        train_ids=sp.loc[sp[col].eq('train'),'segment_id']
        candidates=['HL02','HL03','HL04','HL05','HL06']+exog
        true_constants=[c for c in candidates if train.loc[train.segment_id.isin(train_ids),c].nunique()<=1]
        old_constants=[c for c in candidates if train.loc[train.segment_id.isin(legacy_ids),c].nunique()<=1]
        emit('constant_selector',{'tag':tag,'fold':fold,'extraneous_segments':len(set(legacy_ids)-set(train_ids)),'actual_constants':true_constants,'legacy_constants':old_constants})
        for flag in ['train','val','test']:
            ds=quiet(Dataset_Custom,**common,data_path=f'train_{tag}.csv',split_file=f'dataset/splits_train_{tag}.csv',fold=fold,flag=flag,timeenc=1)
            selected=sp['split'].eq('test') if flag=='test' else sp[col].eq(flag)
            expected=int(sp.loc[selected,'n_windows'].sum())
            ids=sp.loc[sp[col].eq('train'),'segment_id']
            clean=train.segment_id.isin(ids)&train[candidates+['HL01']].notna().all(axis=1)
            mean=float(train.loc[clean,'HL01'].mean())
            emit('loader',{'tag':tag,'fold':fold,'flag':flag,'actual':len(ds),'expected':expected,'match':len(ds)==expected,'scaler_y_mean':float(ds.scaler_y.mean_[0]),'actual_train_mean':mean,'mean_match':bool(np.isclose(ds.scaler_y.mean_[0],mean)),'bool_cols':ds.bool_x_cols})
    a=SimpleNamespace(**{k:v for k,v in common.items() if k not in ['size','model_name']},data='custom',model='DLinearMix2',data_path=f'train_{tag}.csv',split_file=f'dataset/splits_train_{tag}.csv',fold=1,seq_len=96,label_len=30,pred_len=15,embed='timeF',train_only=False,batch_size=64,stride_train=1,stride_eval=1,num_workers=0)
    ds,dl=quiet(data_provider,a,'val')
    torch.manual_seed(42);one=list(iter(dl.sampler));two=list(iter(dl.sampler));keep=(len(ds)//64)*64
    emit('val_sampling',{'tag':tag,'samples':len(ds),'delivered':keep,'dropped':len(ds)-keep,'drop_last':dl.drop_last,'sampler':type(dl.sampler).__name__,'omitted_set_changes':set(one[keep:])!=set(two[keep:])})

# Reproduce row-mode scaler leakage with a 20-row fixture.
fixture=OUT/'row_scaler_fixture.csv'
pd.DataFrame({'date':pd.date_range('2025-01-01',periods=20,freq='min'),'HL01':[0.]*14+[100.]*6}).to_csv(fixture,index=False)
ds=quiet(Dataset_Custom,root_path=str(OUT),data_path=fixture.name,flag='train',size=[3,1,1],target='HL01',model_name='DLinearMix2',input_col='HL01',freq='min',timeenc=1)
emit('row_scaler_leak',{'train_rows':14,'true_train_mean':0,'fitted_mean':ds.scaler_y.mean_.tolist(),'n_samples_seen':int(ds.scaler_y.n_samples_seen_)})

# Tiny pure-function boundary and arithmetic tests.
emit('boundary_exact_need',count_windows_per_segment(pd.DataFrame({'segment_id':[1]*111,'SegmentStart':['2025-01-01']*111,'x':[1.]*111}),seg_col='segment_id',check_cols=['x'],need=111,stride=1).to_dict('records'))
for label,cum,lo,hi,b in [('all_blocked',np.array([0,1,2,3]),1,2,np.array([False,True,True,False])),('single_blocked',np.array([0,1,2]),1,1,np.array([False,True,False]))]:
    try:result=pick_boundary(cum,int(cum[-1]),.5,lo,hi,b)
    except Exception as e:result=type(e).__name__+': '+str(e)
    emit('boundary_probe',{'case':label,'result':result})
m=pd.DataFrame({'segment_id':[1,2,3],'n_windows':[10,0,0]})
emit('empty_eval_partition',assign_split(m,(.7,.15,.15)).to_dict('records'))

for flat in [False,True]:
    for ne in [0,2]:
        torch.manual_seed(42)
        cfg=SimpleNamespace(seq_len=96,pred_len=15,input_col='HL02,HL03',exog_col='rain,gate' if ne else None,branch_in=2,exog_in=ne,mix_in=2+ne,flatten_fusion=flat,dropout=0.)
        model=Model(cfg).eval()
        with torch.no_grad():y=model(torch.zeros(2,96,2+ne))
        emit('model_shape',{'flatten':flat,'exog_in':ne,'shape':list(y.shape),'finite':bool(torch.isfinite(y).all())})

ns=extract('exp/exp_Main2.py',['vali','_slice_output','_forward'],cls='Exp_Main')
class Dummy:
    eval=lambda self:None
    train=lambda self:None
    def __call__(self,x):return x
exp=SimpleNamespace(model=Dummy(),device=torch.device('cpu'),use_amp=False,args=SimpleNamespace(features='S',pred_len=2))
exp._forward=lambda x:ns['_forward'](exp,x)
exp._slice_output=lambda p,y:ns['_slice_output'](exp,p,y)
true=np.array([[0,0],[1,1],[2,2],[3,3]],dtype=np.float32)[...,None]
pred=true.copy();pred[:,1]=0
vm=ns['vali'](exp,[(torch.from_numpy(pred),torch.from_numpy(true))])
emit('corr_zero_variance',{'vali':vm,'test_corr_per_horizon':CORR(pred,true).tolist(),'test_corr_mean':float(CORR(pred,true).mean())})

# Compare two actual merge functions without importing SQL clients.
rename={str(i):c for i,c in enumerate(gates)}
merge=extract('Data_From_SQL_all.py',['merge_all_sources'],env={'GATE_STALENESS_LIMIT':pd.Timedelta('5min'),'GATE_RENAME_MAP':rename,'GATE_TARGET_COLUMNS':list(rename),'DROP_COLS':['StationId']})['merge_all_sources']
grid=pd.DataFrame({'date':pd.date_range('2025-01-01',periods=8,freq='min')})
gate=pd.DataFrame({'date':[grid.date[0],grid.date[2]],**{c:[-1.,np.nan] if i%2 else [1.,2.] for i,c in enumerate(gates)}})
one=merge(grid,grid.copy(),gate)
two=merge_gate_per_column(grid,gate,gates,pd.Timedelta('5min'))
emit('merge_equivalence',{'same':bool(np.isclose(one[gates],two[gates],equal_nan=True).all()),'age5_valid':bool(one.loc[5,gates[1]]==0),'age6_nan':bool(pd.isna(one.loc[6,gates[1]]))})

# T-04 audit reports only presence/category/line; never secrets.
tree=ast.parse(Path('Data_From_SQL_4.py').read_text())
for node in ast.walk(tree):
    if isinstance(node,ast.Assign):
        names=[t.id for t in node.targets if isinstance(t,ast.Name)]
        if any('PASSWORD' in n.upper() or 'SERVER' in n.upper() or 'USER' in n.upper() for n in names):
            emit('db_config_redacted',{'names':names,'line':node.lineno,'literal_string':isinstance(node.value,ast.Constant) and isinstance(node.value.value,str),'reads_environment':'getenv' in ast.unparse(node.value)})

print('PASS: audit finished without calling train, test, run.py main, or checkpoint methods.')
