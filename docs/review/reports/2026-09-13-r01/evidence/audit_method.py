"""Audit evaluation populations and claims with no training or model inference."""
import sys,json,contextlib,io
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path.cwd()))
from data_provider.Data_Loader import Dataset_Custom
from build_drycut_segments_meta import find_drycut_core_segments
OUT=Path('docs/review/reports/2026-09-13-r01/evidence')
def emit(k,v):print(k,json.dumps(v,ensure_ascii=False,default=str))
def quiet(fn,*a,**kw):
    with contextlib.redirect_stdout(io.StringIO()):return fn(*a,**kw)
origins={}
for tag in ['drycut_L3h_buf60','old']:
    d=pd.read_csv(f'dataset/train_{tag}.csv',parse_dates=['date']); sp=pd.read_csv(f'dataset/splits_train_{tag}.csv')
    gates=[c for c in d if 'gate_opening' in c and c!='north_gate_opening_4']
    for mode,extra in [('msr',['min_since_rain','Past10Min','Past1Hr','Now']+gates),('isRain',['isRain','Past10Min','Past1Hr','Now']+gates),('no_exog',[])]:
        ds=quiet(Dataset_Custom,root_path='dataset',data_path=f'train_{tag}.csv',flag='test',size=[96,30,15],input_col='HL02,HL03,HL04,HL05,HL06',exog_col=','.join(extra) or None,segment_col='segment_id',target='HL01',model_name='DLinearMix2',split_file=f'dataset/splits_train_{tag}.csv',freq='min',timeenc=1)
        starts=ds.valid_starts
        origin=pd.to_datetime(ds.dates[starts+95])
        truth=ds.y_raw[starts[:,None]+96+np.arange(15)[None,:]]
        persist=np.repeat(ds.y_raw[starts+95,None],15,axis=1)
        origins[tag,mode]=set(origin)
        corr=[float(np.corrcoef(persist[:,h],truth[:,h])[0,1]) for h in range(15)]
        emit('population',{'tag':tag,'exog':mode,'test_windows':len(ds),'unique_origins':len(set(origin)),'origin_min':origin.min(),'origin_max':origin.max(),'test_persistence_MSE':float(np.mean((truth-persist)**2)),'test_persistence_corr_mean':float(np.mean(corr)),'bool_cols':ds.bool_x_cols})
    test_ids=sp.loc[sp.split.eq('test'),'segment_id']
    emit('seasonality',{'tag':tag,'split':{name:{'past10_wet_fraction':float(d.loc[d.segment_id.isin(sp.loc[sp.split.eq(name),'segment_id']),'Past10Min'].gt(0).mean()),'past10_p95':float(d.loc[d.segment_id.isin(sp.loc[sp.split.eq(name),'segment_id']),'Past10Min'].quantile(.95))} for name in ['train','val','test']}})
for mode in ['msr','isRain','no_exog']:
    a=origins['drycut_L3h_buf60',mode];b=origins['old',mode]
    emit('common_test_origins',{'exog':mode,'intersection':len(a&b),'drycut_only':len(a-b),'old_only':len(b-a)})
for tag in ['drycut_L3h_buf60','old']:
    emit('exog_test_population_change',{'tag':tag,'no_exog_extra':len(origins[tag,'no_exog']-origins[tag,'msr'])})
a=pd.read_csv('dataset/train_drycut_L3h_buf60.csv').drop_duplicates('date').set_index('date')
b=pd.read_csv('dataset/train_old.csv').drop_duplicates('date').set_index('date')
ix=a.index.intersection(b.index); gates=[c for c in a if 'gate_opening' in c]
x=a.loc[ix,gates];y=b.loc[ix,gates]
emit('shared_date_gate_comparison',{'dates':len(ix),'unequal_cells_including_NaN':int((~np.isclose(x,y,equal_nan=True)).sum()),'both_nonnull_unequal':int(((~np.isclose(x,y))&x.notna()&y.notna()).sum().sum())})

level=np.repeat(np.arange(5)*1000.,20);change=np.tile(np.arange(20.),5)
truth=level+change; pred=level-change
emit('pooled_counterexample',{'pooled_corr':float(np.corrcoef(pred,truth)[0,1]),'each_segment_corr':[-1.]*5,'description':'equal event level, reverse within-event slope'})

# Same rain history up to t=115; only future rain differs, isRain at t=115 changes.
rain=np.zeros(400);rain[100:110]=.5
rain_future=rain.copy();rain_future[130:140]=.5
date=pd.date_range('2025-01-01',periods=400,freq='min')
flags=[]
for values in [rain,rain_future]:
    meta=find_drycut_core_segments(pd.DataFrame({'date':date,'rain':values}),rain_col='rain',l_minutes=180)
    flags.append(bool(((meta.SegmentStart<=date[115])&(meta.SegmentEnd>=date[115])).any()))
emit('isRain_future_counterexample',{'same_history_through':date[115],'future_changes_at':date[130],'core_without_future_rain':flags[0],'core_with_future_rain':flags[1],'both_within_retained_buffer':True})

inventory=json.loads((OUT/'run_inventory.json').read_text())
from collections import Counter
emit('run_counts',{'all_roots':inventory['count'],'by_root':dict(Counter(r['path'].split('/')[0] for r in inventory['rows'])),'HL01_in_input':sum('HL01' in (r['input_col'] or '').split(',') for r in inventory['rows'])})
for r in inventory['rows']:
    if '202609' in r['path']:
        p=Path(r['path']).parent
        arrays={}
        for fn in ['pred.npy','true.npy','persist.npy']:
            q=p/'outputs'/fn
            if q.exists():
                z=np.load(q,mmap_mode='r');arrays[fn]={'shape':list(z.shape),'finite':bool(np.isfinite(z).all())}
        emit('smoke_artifact',{'run_id':r['run_id'],'split_mode':r['split_mode'],'fold':r['fold'],'arrays':arrays})
for p in Path('runs').rglob('anchored_metrics.json'):
    d=json.loads(p.read_text())
    emit('historical_anchored_metadata',{'path':str(p),'metadata':d})
