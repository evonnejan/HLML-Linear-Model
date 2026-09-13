"""Read-only audit; run from repo root with .HLML_Linear_venv/bin/python -B.
All outputs go to stdout. Large CSVs use chunks/usecols; never invoke training.
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path.cwd()))
from build_drycut_segments_meta import find_drycut_core_segments
from build_training_csv_from_meta import add_min_since_rain, fill_gate_within_segment
from build_splits import count_windows_per_segment, find_overlap_groups

def emit(key, obj):
    print(key, json.dumps(obj, ensure_ascii=False, default=str))

emit('environment', {'python':sys.version,'pandas':pd.__version__,'numpy':np.__version__})
for p in sorted(Path('dataset').glob('*')):
    if p.suffix != '.csv': continue
    h=hashlib.sha256(); n=0; last_byte=b''
    with p.open('rb') as f:
        bom=f.read(3)==b'\xef\xbb\xbf'; f.seek(0)
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b); n+=b.count(b'\n'); last_byte=b[-1:]
    emit('inventory', {'file':p.name,'bytes':p.stat().st_size,'newline_count':n,'rows':n+int(last_byte!=b'\n')-1,'columns':list(pd.read_csv(p,nrows=0).columns),'bom':bom,'sha256':h.hexdigest()})

p=Path('dataset/all_minute_wide.csv')
cols=list(pd.read_csv(p,nrows=0).columns); gates=[c for c in cols if 'gate_opening' in c]
rows=0; bad_grid=0; prev=None; na=pd.Series(0,index=cols,dtype='int64'); gate_na=0; quant={}; parts=[]
for c in pd.read_csv(p,chunksize=50000,parse_dates=['date']):
    d=c.date.diff().dropna(); bad_grid+=int((d!=pd.Timedelta('1min')).sum())
    if prev is not None: bad_grid+=int(c.date.iloc[0]-prev!=pd.Timedelta('1min'))
    prev=c.date.iloc[-1]; rows+=len(c); na+=c.isna().sum(); gate_na+=int(c[gates].isna().any(axis=1).sum())
    for col in ['Past10Min','Past1Hr','Now']:
        x=c[col].dropna(); q=quant.setdefault(col,{'wet':0,'off_half_mm':0,'min_positive':float('inf')})
        q['wet']+=int((x>0).sum()); q['off_half_mm']+=int((~np.isclose(x*2,np.round(x*2))).sum())
        if (x>0).any(): q['min_positive']=min(q['min_positive'],float(x[x>0].min()))
    parts.append(c[['date','Past10Min','Past1Hr']])
emit('wide_stats',{'rows':rows,'bad_grid':bad_grid,'last':prev,'na_counts':na.to_dict(),'gate_any_na_pct':100*gate_na/rows,'quantization':quant})
rain=pd.concat(parts,ignore_index=True); del parts
rain=add_min_since_rain(rain)
for rc in ['Past10Min','Past1Hr']:
    m=find_drycut_core_segments(rain,rain_col=rc,l_minutes=180)
    existing=pd.read_csv('dataset/rain_segments_meta_drycut_L3h_buf60.csv',parse_dates=['SegmentStart','SegmentEnd'])
    emit('drycut_reconstruction',{'rain_col':rc,'segments':len(m),'core_matches':m[['SegmentStart','SegmentEnd']].equals(existing[['SegmentStart','SegmentEnd']]),'wet_minutes':int(m.WetMinutes.sum()),'length_usable':{b:int((((m.SegmentEnd-m.SegmentStart).dt.total_seconds()/60+1+2*b)>=111).sum()) for b in [0,30,60]}})

both=same=0; old_na=0; newer_na=0
for a,b in zip(pd.read_csv('dataset/all_minute_wide.gatev1.bak.csv',usecols=['date']+gates,chunksize=50000),pd.read_csv(p,usecols=['date']+gates,chunksize=50000)):
    assert a.date.equals(b.date)
    mask=a[gates].notna() & b[gates].notna(); both+=int(mask.sum().sum())
    same+=int((np.isclose(a[gates],b[gates]) & mask).sum().sum())
    old_na+=int(a[gates].isna().any(axis=1).sum()); newer_na+=int(b[gates].isna().any(axis=1).sum())
emit('gate_before_after',{'both':both,'isclose':same,'before_any_na_pct':100*old_na/rows,'after_any_na_pct':100*newer_na/rows})

wide_gate=pd.read_csv(p,usecols=['date']+gates,parse_dates=['date']).set_index('date')
for tag in ['drycut_L3h_buf60','old']:
    train=pd.read_csv(f'dataset/train_{tag}.csv',parse_dates=['date','SegmentStart','SegmentEnd','WinStart','WinEnd'])
    sp=pd.read_csv(f'dataset/splits_train_{tag}.csv').fillna('')
    sizes=train.groupby('segment_id').size()
    msr=train.min_since_rain
    expected=rain.set_index('date').loc[train.date,'min_since_rain'].to_numpy()
    base=['HL01','HL02','HL03','HL04','HL05','HL06','Past10Min','Past1Hr','Now']+gates
    windows={}
    for extra in ['min_since_rain','isRain','none','no_exog']:
        ck=base+([extra] if extra not in ['none','no_exog'] else [])
        if extra=='no_exog': ck=base[:6]
        cnt=count_windows_per_segment(train,seg_col='segment_id',check_cols=ck,need=111,stride=1)
        windows[extra]={'total':int(cnt.n_windows.sum()),'dead_segments':int(cnt.n_windows.eq(0).sum()),'split_mismatches':int((cnt.set_index('segment_id').n_windows-sp.set_index('segment_id').n_windows).ne(0).sum())}
    meta=train[['segment_id','WinStart','WinEnd']].drop_duplicates().sort_values('WinStart')
    groups=find_overlap_groups(meta,'segment_id')
    crossing={c:sum(sp.set_index('segment_id').loc[g,c].nunique()>1 for g in groups) for c in ['split','fold_1','fold_2','fold_3']}
    gap=(meta.WinStart.shift(-1)-meta.WinEnd).dt.total_seconds()/60
    sub=wide_gate.loc[train.date].reset_index(); sub['segment_id']=train.segment_id.to_numpy()
    raw_na=sub[gates].isna()
    # age since last non-null minute within a segment: lower bound on physical observation age.
    ages=pd.DataFrame(index=sub.index)
    for c in gates:
        last=sub.date.where(sub[c].notna()).groupby(sub.segment_id).ffill()
        ages[c]=(sub.date-last).dt.total_seconds()/60
    filled=fill_gate_within_segment(sub)
    emit('training_stats',{'tag':tag,'shape':train.shape,'segments':len(sizes),'length_min_med_max':[int(sizes.min()),float(sizes.median()),int(sizes.max())],'date_range':[train.date.min(),train.date.max()],'isRain':train.isRain.value_counts().to_dict(),'isRain_formula_mismatch':int((train.isRain.ne(train.date.between(train.SegmentStart,train.SegmentEnd))).sum()),'msr_min_med_max':[msr.min(),msr.median(),msr.max()],'msr_na':int(msr.isna().sum()),'msr_formula_mismatch':int((~np.isclose(msr,expected,equal_nan=True)).sum()),'msr_nan_segments':train.loc[msr.isna()].groupby('segment_id').size().to_dict(),'msr_max_examples':train.loc[msr.eq(msr.max()),['segment_id','date','SegmentStart','Past10Min','Past1Hr']].to_dict('records'),'gate_any_na_pct':float(train[gates].isna().any(axis=1).mean()*100),'gate_ffill_mismatches':int((~np.isclose(filled[gates],train[gates],equal_nan=True)).sum()),'gate_fill_age_over5_cells':int(((ages>5)&raw_na).sum().sum()),'gate_fill_max_age_lower_bound_min':float(ages.where(raw_na).max().max()),'windows':windows,'overlap_groups':len(groups),'group_segments':sum(map(len,groups)),'crossing_groups':crossing,'min_gap':gap.min(),'duplicate_minutes':int(train.date.duplicated().sum()),'within_segment_bad_grid':int(train.groupby('segment_id').date.diff().dropna().ne(pd.Timedelta('1min')).sum())})
    for c in ['split','fold_1','fold_2','fold_3']:
        emit('split_stats',{'tag':tag,'column':c,'groups':sp.groupby(c).agg(segments=('segment_id','size'),windows=('n_windows','sum')).to_dict(),'periods':sp.groupby(c).SegmentStart.agg(['min','max']).to_dict()})
    if tag=='drycut_L3h_buf60':
        maximum=train.loc[msr.idxmax(),'date']; last=rain.loc[(rain.date<=maximum)&(rain.Past10Min>0),'date'].max()
        emit('msr_max_cause',{'date':maximum,'last_positive_past10min':last,'difference_minutes':(maximum-last).total_seconds()/60})

gate=pd.read_csv('dataset/wra_cogate_obs_wide_gate_opening.csv',parse_dates=['date'])
emit('raw_gate',{'rows':len(gate),'partial_rows':int(gate[gates].notna().sum(axis=1).between(1,6).sum()),'negative_counts':gate[gates].lt(0).sum().to_dict(),'minima':gate[gates].min().to_dict(),'interval_quantiles_minutes':{c:((gate.loc[gate[c].notna(),'date'].sort_values().diff().dt.total_seconds()/60).quantile([.5,.9,.99,1]).to_dict()) for c in gates}})
