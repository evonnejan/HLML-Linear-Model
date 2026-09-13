import sys,io,contextlib,json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path.cwd()))
from build_splits import assign_split,find_overlap_groups,blocked_boundaries
from data_provider.Data_Loader import Dataset_Custom
def emit(k,v):print(k,json.dumps(v,ensure_ascii=False,default=str))
m=pd.DataFrame({'segment_id':range(1,7),'n_windows':[1,1,7,1,1,1]})
b=blocked_boundaries(m,[[4,5,6]],'segment_id')
try:r=assign_split(m,(.7,.15,.15),b).to_dict('records')
except Exception as e:r=str(e)
emit('greedy_split_false_infeasible',{'windows':m.n_windows.tolist(),'blocked':b.tolist(),'result':r,'safe_alternative_boundaries':[2,3],'safe_partition_windows':[2,7,3]})

gates=[c for c in pd.read_csv('dataset/all_minute_wide.csv',nrows=0).columns if 'gate_opening' in c]
wide=pd.read_csv('dataset/all_minute_wide.csv',usecols=['date']+gates,parse_dates=['date']).set_index('date')
for tag in ['drycut_L3h_buf60','old']:
    d=pd.read_csv(f'dataset/train_{tag}.csv',parse_dates=['date'])
    src=wide.loc[d.date].reset_index();src['segment_id']=d.segment_id.to_numpy()
    ages=pd.DataFrame(index=src.index)
    for c in gates:
        last=src.date.where(src[c].notna()).groupby(src.segment_id).ffill()
        ages[c]=(src.date-last).dt.total_seconds()/60
    max_in_input=0.; windows_stale=0; anchors_ok=True
    for flag in ['train','val','test']:
        with contextlib.redirect_stdout(io.StringIO()):
            ds=Dataset_Custom(root_path='dataset',data_path=f'train_{tag}.csv',flag=flag,size=[96,30,15],input_col='HL02,HL03,HL04,HL05,HL06',exog_col=','.join(['min_since_rain','Past10Min','Past1Hr','Now']+gates),segment_col='segment_id',target='HL01',model_name='DLinearMix2',split_file=f'dataset/splits_train_{tag}.csv',freq='min',timeenc=1)
        # Resolve segment-aware row positions, including duplicate dates in old.
        sp=pd.read_csv(f'dataset/splits_train_{tag}.csv')
        ids=sp.loc[sp.split.eq(flag),'segment_id']
        subset=d.loc[d.segment_id.isin(ids)].sort_values(['segment_id','date'])
        a=ages.loc[subset.index].to_numpy()
        selected=np.zeros(len(a),dtype=bool)
        row_stale=np.nan_to_num(a,nan=0).max(axis=1)>5
        cs=np.r_[0,np.cumsum(row_stale)]
        windows_stale+=int(((cs[ds.valid_starts+96]-cs[ds.valid_starts])>0).sum())
        for s in ds.valid_starts:selected[s:s+96]=True
        max_in_input=max(max_in_input,float(np.nanmax(a[selected])))
        for i in np.linspace(0,len(ds)-1,min(5,len(ds)),dtype=int):
            x,y,*_=ds[int(i)]; actual=float(ds.inverse_transform(y[29:30])[0,0]); expected=float(ds.y_raw[ds.valid_starts[i]+95]); anchors_ok &= bool(np.isclose(actual,expected))
    emit('stale_values_reach_valid_model_inputs',{'tag':tag,'windows_with_age_gt5_input':windows_stale,'max_age_lower_bound_in_input':max_in_input,'sampled_label_anchors_correct':anchors_ok})

print('No training; no model checkpoint or output directory was accessed for writing.')
