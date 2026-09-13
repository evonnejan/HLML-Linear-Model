"""Read-only documentation inventory and reference probes."""
import ast,hashlib,json,subprocess
from pathlib import Path
OUT=Path('docs/review/reports/2026-09-13-r01/evidence')
def emit(k,v):print(k,json.dumps(v,ensure_ascii=False,default=str))
core=['Data_From_SQL_all.py','rebuild_gate_columns.py','build_drycut_segments_meta.py','build_training_csv_from_meta.py','build_splits.py','data_provider/Data_Loader.py','data_provider/Data_Factory.py','run.py','models/DLinearMix2.py','exp/exp_Main2.py','exp/exp_Basic.py','utils/metrics.py']
for p in core:
    raw=Path(p).read_bytes();tree=ast.parse(raw.decode('utf-8-sig'))
    deps=[{'line':n.lineno,'module':n.module,'names':[a.name for a in n.names]} for n in tree.body if isinstance(n,ast.ImportFrom)]
    emit('core',{'file':p,'lines':len(raw.splitlines()),'sha256':hashlib.sha256(raw).hexdigest(),'imports':deps})
for p in sorted(Path('docs/review').rglob('*.md')):
    if '2026-09-13-r01' in str(p):continue
    log=subprocess.check_output(['git','log','-1','--format=%h %aI','--',str(p)],text=True).strip()
    emit('doc',{'file':str(p),'lines':len(p.read_text().splitlines()),'last_commit':log})
checks=[('run.py',169,174),('run.py',119,156),('models/DLinearMix2.py',151,158),('models/DLinearMix2.py',144,165),('data_provider/Data_Loader.py',247,250),('data_provider/Data_Loader.py',277,277),('data_provider/Data_Loader.py',307,311),('data_provider/Data_Loader.py',457,472),('data_provider/Data_Loader.py',474,489),('exp/exp_Main2.py',547,547),('build_drycut_segments_meta.py',91,92),('build_drycut_segments_meta.py',106,107),('build_splits.py',143,149),('exp/exp_Main2.py',184,221),('exp/exp_Main2.py',696,880)]
for f,a,b in checks:
    t=Path(f).read_text().splitlines()
    tree=ast.parse('\n'.join(t))
    functions=[n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.lineno<=a<=n.end_lineno]
    emit('reference',{'file':f,'range':[a,b],'function_at_start':functions,'lines':[f'{i}: {t[i-1]}' for i in range(a,min(b,a+5)+1)]})
files=list(Path('dataset').iterdir())
emit('dataset_file_counts',{'csv':len([p for p in files if p.suffix=='.csv']),'sha256_sidecars':len([p for p in files if p.suffix=='.sha256']),'other':[p.name for p in files if p.suffix not in ['.csv','.sha256']]})
emit('license_files',subprocess.run(['rg','--files','--hidden','-g','*LICENSE*','-g','*NOTICE*','-g','!*venv*/**','-g','!.git/**'],capture_output=True,text=True).stdout.splitlines())
all_names=subprocess.run(['rg','--files','dataset','docs'],capture_output=True,text=True).stdout.splitlines()
emit('station_metadata_candidates', [p for p in all_names if any(s in Path(p).name.lower() for s in ['station','location','coordinate','測站','字典'])])
