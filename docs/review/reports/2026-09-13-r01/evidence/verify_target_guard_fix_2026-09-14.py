"""Verify the specifically authorized C-01 change, without training or imports."""
from pathlib import Path
from datetime import datetime
import ast
import hashlib
import json
import re
import subprocess

REPORT = Path("docs/review/reports/2026-09-13-r01")
EVIDENCE = REPORT / "evidence"

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def records(file, prefix):
    return [json.loads(line[len(prefix)+1:]) for line in (EVIDENCE/file).read_text().splitlines()
            if line.startswith(prefix+" ")]

print("Checked", datetime.now().astimezone().isoformat(timespec="seconds"))
changed = {"run.py", "data_provider/Data_Loader.py"}
cores = records("docs.txt", "core")
for entry in cores:
    same = digest(Path(entry["file"])) == entry["sha256"]
    assert same == (entry["file"] not in changed), entry["file"]
    ast.parse(Path(entry["file"]).read_bytes())
print("PASS: exactly two audited core files changed; ten others match the original hashes.")
csvs = records("data.txt", "inventory")
assert {p.name for p in Path("dataset").glob("*.csv")} == {r["file"] for r in csvs}
for entry in csvs:
    assert digest(Path("dataset")/entry["file"]) == entry["sha256"]
print("PASS: all 20 dataset CSV hashes unchanged (streaming verification).")

tree = ast.parse((EVIDENCE/"audit_completeness.py").read_text())
base_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "BASELINE" for t in n.targets))
baseline = json.loads(ast.literal_eval(base_node.value.args[0]))
for name, expected in baseline["evidence"].items():
    assert digest(EVIDENCE/name) == expected
report = (REPORT/"report.md").read_text()
bodies = re.findall(r"^#### [DCMR]-\d+.*?(?=^#### |^### |^## |\Z)", report, re.M|re.S)
assert len(bodies) == 24
assert hashlib.sha256("\n".join(bodies).encode()).hexdigest() == baseline["findings_sha256"]
assert len(re.findall(r"^\| T-\d+ \|", report, re.M)) == 33
assert len(re.findall(r"^### Q-[DCMR]\d+ ·", report, re.M)) == 11
assert "第5d節" in report and "其餘9題" in report
print("PASS: original 24 finding bodies, 33 T rows, 11 historical questions and 15 evidence files preserved.")
assert "24 passed" in (EVIDENCE/"target_guard_tests_2026-09-14.txt").read_text()
ast.parse(Path("tests/test_target_input_guard.py").read_text())
print("PASS: targeted regression run reports 24 passing cases.")

for name in ["report.md", "CHANGES.md"]:
    for link in re.findall(r"\]\(([^)]+)\)", (REPORT/name).read_text()):
        if not link.startswith(("https://", "http://", "#")):
            assert (REPORT/link.split("#", 1)[0]).exists(), link
tracked = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
assert set(tracked) == changed
assert not subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).strip()
subprocess.run(["git", "-c", "core.whitespace=cr-at-eol", "diff", "--check"], check=True)
status = subprocess.check_output(["git", "status", "--short"], text=True)
assert {line[3:] for line in status.splitlines()} == changed | {
    str(REPORT)+"/", "tests/test_target_input_guard.py"}
raw = Path("run.py").read_bytes()
assert raw.count(b"\r\n") == raw.count(b"\n")
assert not Path("data_provider/Data_Loader.py").read_bytes().endswith(b"\n")
print("PASS: scope, no staged changes, CRLF-aware diff check, and original newline formats.")
print(status, end="")
print("Current source hashes", json.dumps({p:digest(Path(p)) for p in sorted(changed | {"tests/test_target_input_guard.py"})}))
print("Tracked patch follows (new test is separately stored at tests/test_target_input_guard.py):")
print(subprocess.check_output(["git", "diff", "--", *sorted(changed)], text=True))
print("PASS: no training, model forward, checkpoint access, commit, or push in this verification.")
