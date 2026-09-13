"""Read-only completion audit. No project imports, training, or dataset writes.
Run from repository root with python3 -B; save stdout only in this report folder.
Original evidence and findings are compared to the pre-completion-edit snapshot.
"""
from pathlib import Path
from datetime import datetime
from collections import Counter
import hashlib, json, re, subprocess

ROOT = Path("docs/review/reports/2026-09-13-r01")
BASELINE = json.loads("{\"evidence\":{\"audit_supplement.py\":\"981e08c26d456c5d866f03f107b5e1f1364c58e8360a471ef2ffe50069de56d8\",\"audit_docs.py\":\"80bbd8079f1c3e4fc9154e5f0f868d415411184dd2e13d14e8861b306e786520\",\"audit_code.py\":\"bdc71c6d1d6f44e21bc2f16cd00b957d4e4dd396974540d97323b5b8a05210e6\",\"method.txt\":\"baeeb1859540168c5644a194a46a6b9f1406a03df122e14a8fcd17af7c509d2b\",\"row_scaler_fixture.csv\":\"9718e7ebac6ab52fa3562da821c3e62231950b826d312d772119f711e91468fa\",\"commands.md\":\"75771cf079bf0bf4006e7d350bb07fe368d654d660487f80bb3d0314e0542e96\",\"audit_method.py\":\"8c21d66317a1ee21322a63cd87409712574f4f0339e4ffb242d7fe46a08bc4b9\",\"code.txt\":\"8d5fe56f7e00650ee259bd31f140978b3197a50360acf27e75c15fbc770cc47f\",\"verification.txt\":\"7f1b325e55a740678e96a2847985d313e6154300bec27c5628c06ac9c6cc40a5\",\"supplement.txt\":\"e3375e3a383bb54d8c9eaf257bdf316f096d8c429e39109fb3fb4caf0372564f\",\"cli_effects.txt\":\"7ba1ba7eff76b990851f22cca6df3df916cb9fa4d99a45f10c4612d3d01c44c9\",\"run_inventory.json\":\"32c9933c01eb839c30df339ca03ed02968c548e56501cafbae92c6f7f69f3f1d\",\"audit_data.py\":\"8ba87c1905520378b25521c67267e0a86497daa1a6ff193e5706146ecaa80a65\",\"data.txt\":\"f81fb3b0b8f42f7c1c205421e442a4c550494e6811c539079ed9147ffec4a6a8\",\"docs.txt\":\"86b15281280d3800fa4046987caabe66e85f027f7a0cd2d3ac7976816f9a2531\"},\"findings_sha256\":\"99162937de606149c1bc50a3f3dc66a7effc762190c7ad781c89b2cc3a98fc01\"}")

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def records(name, prefix):
    for line in (ROOT / "evidence" / name).read_text().splitlines():
        if line.startswith(prefix + " "):
            yield json.loads(line[len(prefix) + 1:])

def emit(kind, value):
    print(kind, json.dumps(value, ensure_ascii=False), flush=True)

emit("checked", datetime.now().astimezone().isoformat(timespec="seconds"))
for name, expected in BASELINE["evidence"].items():
    assert digest(ROOT / "evidence" / name) == expected, name
emit("original_evidence", {"unchanged": len(BASELINE["evidence"])})

cores = list(records("docs.txt", "core"))
assert len(cores) == 12
for row in cores:
    assert digest(Path(row["file"])) == row["sha256"], row["file"]
emit("core_hashes", {"unchanged": len(cores)})

csvs = list(records("data.txt", "inventory"))
assert len(csvs) == 20
assert {p.name for p in Path("dataset").glob("*.csv")} == {r["file"] for r in csvs}
for row in csvs:
    assert digest(Path("dataset") / row["file"]) == row["sha256"], row["file"]
emit("dataset_hashes", {"unchanged": len(csvs), "method": "streaming 1 MiB chunks"})

report = (ROOT / "report.md").read_text()
sections = re.findall(r"^#### ([DCMR]-\d+).*?(?=^#### |^### |^## |\Z)", report, re.M | re.S)
bodies = re.findall(r"^#### [DCMR]-\d+.*?(?=^#### |^### |^## |\Z)", report, re.M | re.S)
assert len(sections) == len(set(sections)) == 24
assert hashlib.sha256("\n".join(bodies).encode()).hexdigest() == BASELINE["findings_sha256"]
counts = Counter()
for block in bodies:
    for field in ["區塊", "子類", "嚴重度", "位置", "現象", "為什麼是問題", "建議", "信心度", "證據"]:
        assert f"**{field}：**" in block, (block.splitlines()[0], field)
    counts[re.search(r"\*\*嚴重度：\*\*\s*(Blocker|Major|Minor|Nit)", block).group(1)] += 1
assert counts == {"Blocker": 1, "Major": 17, "Minor": 6}
tids = re.findall(r"^\| (T-\d+) \|", report, re.M)
assert tids == [f"T-{i:02}" for i in range(1, 34)]
qids = re.findall(r"^### (Q-[DCMR]\d+) ·", report, re.M)
assert len(qids) == len(set(qids)) == 11
blocks = re.split(r"^## [1-4]\. Block [DCMR].*$", report, flags=re.M)[1:]
claims = []
for block in blocks:
    claim = re.search(r"^### 本 block.*?\n(.*?)(?=^### )", block, re.M | re.S).group(1)
    count = len(re.findall(r"^\d+\. ", claim, re.M))
    assert 5 <= count <= 10
    claims.append(count)
emit("report_schema", {"findings": 24, "unchanged_finding_bodies": True,
                       "severity": dict(counts), "T_rows": len(tids),
                       "pending_questions": len(qids), "claims_D_C_M_R": claims})

links = 0
for name in ["report.md", "CHANGES.md"]:
    content = (ROOT / name).read_text()
    for target in re.findall(r"\]\(([^)]+)\)", content):
        if target.startswith(("https://", "http://", "#")):
            continue
        assert (ROOT / target.split("#", 1)[0]).exists(), target
        links += 1
emit("local_markdown_links", {"valid": links})

head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert head == "fd2903764d6e6f4f6334d2ad71048d1293d40a24"
tracked = subprocess.check_output(["git", "diff", "HEAD", "--stat"], text=True).strip()
assert not tracked, tracked
status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
emit("git", {"HEAD": head, "tracked_diff": tracked or "(empty)", "status": status})
assert status == "?? docs/review/reports/2026-09-13-r01/", status
emit("artifacts", {str(p.relative_to(ROOT)): digest(p)
                   for p in sorted(ROOT.rglob("*"))
                   if p.is_file() and p.name != "completeness.txt"})
print("PASS: completion audit; original evidence and finding bodies unchanged; no project code imported.")

