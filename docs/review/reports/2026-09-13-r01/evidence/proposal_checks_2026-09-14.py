"""In-memory probes for the user's proposed fixes. No training or production writes."""
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import pandas as pd
import run_matrix as rm
from utils.metrics import CORR

BASE = Path('docs/review/reports/2026-09-13-r01')
ATTACHMENT = Path('/Users/zkc/.codex/attachments/90e81b5c-ef69-4376-b75c-36fc6edd955c/pasted-text.txt')
paths = ['run_matrix.py', 'collect_matrix.py', 'run.py', 'exp/exp_Main2.py',
         'utils/metrics.py', 'data_provider/Data_Factory.py', 'data_provider/Data_Loader.py',
         'build_splits.py', 'compute_anchored_mse.py', 'experiments/manifest.csv',
         'dataset/splits_train_old.csv', 'dataset/splits_train_old.json',
         'dataset/splits_train_drycut_L3h_buf60.csv', 'dataset/splits_train_drycut_L3h_buf60.json']
hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
result = {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'source_hashes': hashes, 'pandas': pd.__version__,
          'attachment_sha256': hashlib.sha256(ATTACHMENT.read_bytes()).hexdigest()}

# Copy exactly the proposed schema/read policy, with the real pending manifest.
dtypes = {c: 'string' for c in ['run_key', 'dataset', 'exog', 'loss', 'status',
                              'run_dir', 'started', 'finished', 'error']}
dtypes.update(fold='Int64', duration_s='Float64')
df = pd.read_csv('experiments/manifest.csv', dtype=dtypes,
                 keep_default_na=False, na_values=[''])
assert str(df.duration_s.dtype) == 'Float64'
df.at[0, 'duration_s'] = 135.7
result['proposed_read'] = {'duration_dtype': str(df.duration_s.dtype),
                          'assign_135_7': float(df.at[0, 'duration_s']),
                          'missing_text': {c: bool(pd.isna(df.at[0, c])) for c in ['run_dir', 'error', 'finished']}}
for name, fn in [('path_from_missing_run_dir', lambda: Path(df.at[0, 'run_dir'])),
                 ('slice_missing_error', lambda: df.at[0, 'error'][:160]),
                 ('truth_test_missing_status', lambda: bool(pd.NA == 'done'))]:
    try:
        result[name] = str(fn())
    except Exception as exc:
        result[name] = type(exc).__name__ + ': ' + str(exc)
# In-memory round trip: no tmp CSV or output directories created.
buf = io.StringIO()
df.to_csv(buf, index=False)
buf.seek(0)
again = pd.read_csv(buf, dtype=dtypes, keep_default_na=False, na_values=[''])
assert again.at[0, 'duration_s'] == 135.7
result['numeric_round_trip'] = True
scoped = pd.read_csv('experiments/manifest.csv', dtype=dtypes, keep_default_na=False,
                     na_values={'fold': [''], 'duration_s': ['']})
scoped.at[0, 'duration_s'] = 135.7
assert scoped.at[0, 'error'] == '' and scoped.at[0, 'run_dir'] == ''
result['scoped_numeric_na_policy'] = {'duration_dtype': str(scoped.duration_s.dtype),
                                    'duration_value': float(scoped.at[0, 'duration_s']),
                                    'error_stays_empty_string': scoped.at[0, 'error'] == '',
                                    'run_dir_stays_empty_string': scoped.at[0, 'run_dir'] == ''}

# Proposed hash excludes LOSSES/FOLDS and implicit run.py defaults.
def proposed_hash():
    payload = {'CONST': rm.CONST, 'FLAGS': rm.FLAGS, 'EXOGS': rm.EXOGS, 'DATASETS': rm.DATASETS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
h = proposed_hash()
before = list(rm.FOLDS)
try:
    rm.FOLDS = [1, 2]
    result['fold_change_not_hashed'] = {'before': before, 'after': rm.FOLDS, 'same_hash': proposed_hash() == h}
finally:
    rm.FOLDS = before
before = list(rm.LOSSES)
try:
    rm.LOSSES = ['mse']
    result['loss_grid_change_not_hashed'] = {'before': before, 'after': rm.LOSSES, 'same_hash': proposed_hash() == h}
finally:
    rm.LOSSES = before
tree = ast.parse(Path('run.py').read_text())
default_names = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
        if node.args and isinstance(node.args[0], ast.Constant):
            flag = node.args[0].value
            if flag in ['--huber_beta', '--dlinear_kernel_size', '--fusion_hidden_dim', '--lradj']:
                default = next(k.value.value for k in node.keywords if k.arg == 'default')
                default_names.append({'flag': flag, 'default': default, 'in_CONST': flag[2:] in rm.CONST})
result['implicit_training_defaults_not_hashed'] = default_names

side = json.loads(Path(rm.DATASETS['old']['split_file']).with_suffix('.json').read_text())
csv_name = rm.DATASETS['old']['data_path']
result['data_path_comparison'] = {'sidecar': side['data_path'], 'matrix_data_path': csv_name,
                                'raw_strings_equal': side['data_path'] == csv_name,
                                'resolved_files_equal': Path(side['data_path']).resolve() == (Path('dataset') / csv_name).resolve()}
result['split_values'] = {}
for name, ds in rm.DATASETS.items():
    s = pd.read_csv(ds['split_file'], keep_default_na=False)
    result['split_values'][name] = {c: sorted(s[c].unique().tolist()) for c in ['split', 'fold_1', 'fold_2', 'fold_3']}

exp_tree = ast.parse(Path('exp/exp_Main2.py').read_text())
exp_class = next(n for n in exp_tree.body if isinstance(n, ast.ClassDef) and n.name == 'Exp_Main')
vali = next(n for n in exp_class.body if isinstance(n, ast.FunctionDef) and n.name == 'vali')
train = next(n for n in exp_class.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
def calls(node):
    return sorted(set(ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)))
result['checkpoint_corr_call_path'] = {'vali_calls': calls(vali),
                                     'train_score_lines': [ast.unparse(n) for n in ast.walk(train)
                                         if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ['vali_metrics', 'vali_score', 'alt_score'] for t in n.targets)]}
assert 'np.corrcoef' in calls(vali) and 'CORR' not in calls(vali) and 'metric' not in calls(vali)

tiny = np.array([0., 1e-8, 2e-8])[:, None, None]
result['nonconstant_tiny_scale'] = {'std': float(tiny.std()),
                                  'legacy_corr': float(CORR(tiny, tiny)[0]),
                                  'pearson': float(np.corrcoef(tiny[:, 0, 0], tiny[:, 0, 0])[0, 1])}
expected = {('drycut', 'none', 'mse', 1, 'best_corr', 'raw')}
actual_rows = [next(iter(expected)), next(iter(expected))]
actual = set(actual_rows)
result['set_hides_duplicates'] = {'rows': len(actual_rows), 'unique_rows': len(actual),
                                 'missing': len(expected - actual), 'extra': len(actual - expected)}
guard = compile("assert pred_shape == true_shape", '<proposed_assert>', 'exec', optimize=1)
exec(guard, {'pred_shape': (3, 15, 1), 'true_shape': (1, 15, 1)})
result['optimized_assert_accepts_wrong_shape'] = True
for p, h in hashes.items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h, p
print(json.dumps(result, ensure_ascii=False, indent=2))
