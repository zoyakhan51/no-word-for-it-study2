# study two on kaggle: paste this whole file into one notebook cell
# settings: accelerator GPU T4 x2, internet on, secrets GITHUB_TOKEN (read access to the private repo) and HF_TOKEN
# start it with Save Version then Save and Run All so the secrets reach the run
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from kaggle_secrets import UserSecretsClient

START = time.time()
REPO = "https://github.com/zoyakhan51/no-word-for-it-study2.git"
# pin a commit here once the code is final so the run can be traced to the exact code
COMMIT = "main"
CODE = Path("/tmp/code")
WORK = Path("/kaggle/working")
# no new work starts after this many hours so everything is saved inside the twelve hour limit
HOURS = 11.0
STOP_AT = START+HOURS*3600
SMOKE_ITEMS = 6


def step(title,command,env,cwd=CODE):
    """Run one stage and stop the whole run when it fails"""
    print(f"=== {title} ===",flush=True)
    began = time.time()
    code = subprocess.run(command,env=env,cwd=cwd).returncode
    print(f"{title}: exit code {code} after {round((time.time()-began)/60,1)} min",flush=True)
    assert code==0,f"{title} failed so the run stops here"


secrets = UserSecretsClient()
github_token = secrets.get_secret("GITHUB_TOKEN")
hf_token = secrets.get_secret("HF_TOKEN")
auth = base64.b64encode(f"x-access-token:{github_token}".encode()).decode()
git_env = dict(os.environ,GIT_CONFIG_COUNT="1",GIT_CONFIG_KEY_0="http.extraHeader",GIT_CONFIG_VALUE_0=f"Authorization: Basic {auth}")
if not CODE.exists():
    subprocess.run(["git","clone","--quiet",REPO,str(CODE)],env=git_env,check=True)
subprocess.run(["git","-C",str(CODE),"checkout","--quiet",COMMIT],check=True)
head = subprocess.run(["git","-C",str(CODE),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
print(f"code at commit {head}")
subprocess.run([sys.executable,"-m","pip","install","-q","statsmodels","pytest"],check=True)

# 0 picking up the items and finished chunks of an earlier version when its output is attached as an input
inputs = Path("/kaggle/input")
earlier = [p.parent for p in inputs.rglob("environment_study2.json")] if inputs.exists() else []
assert len(earlier)<=1,f"more than one earlier run is attached so it is unclear which to resume: {earlier}"
if earlier and not (WORK/"results2").exists():
    for name in ("stimuli2","results2"):
        if (earlier[0]/name).exists():
            subprocess.run(["cp","-r",str(earlier[0]/name),str(WORK/name)],check=True)
    print(f"resuming from {earlier[0]}")

base_env = dict(os.environ,PYTHONPATH=str(CODE/"src"),HF_HOME="/tmp/hf",HF_TOKEN=hf_token,
                NWF_STIMULI2=str(WORK/"stimuli2"),NWF_RESULTS2=str(WORK/"results2"),NWF_SCRATCH="/tmp/nwf_scratch")
(WORK/"environment_study2.json").write_text(json.dumps({
    "commit":head,"started":time.strftime("%Y-%m-%d %H:%M:%S"),
    "gpus":subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",shell=True,capture_output=True,text=True).stdout.strip(),
    "packages":subprocess.run("pip list 2>/dev/null | grep -i -E '^(torch|transformers|numpy|scipy|matplotlib|pillow) '",shell=True,capture_output=True,text=True).stdout.strip(),
    "python":sys.version,
},indent=1),encoding="utf-8")

# 1 the unit tests of every new piece on tiny inputs with known answers
step("unit tests",[sys.executable,"-m","pytest","tests/test_s2.py","tests/test_loudness.py","-q","--noconftest","-p","no:cacheprovider"],base_env)

# 2 the fresh study two items built from seeds so they are new to every earlier analysis
if not (WORK/"stimuli2"/"manifest.json").exists():
    step("build stimuli",[sys.executable,"-m","nwf.s2stimuli"],base_env)

# 3 a short gpu run on the first items of every set with checks before the long run starts
smoke_env = dict(base_env,NWF_RESULTS2=str(WORK/"results2_smoke"))
for model in ("qwen","qwen_names"):
    step(f"smoke {model}",[sys.executable,"-m","nwf.s2run","--model",model,"--items",str(SMOKE_ITEMS)],smoke_env)
for model in ("qwen","qwen_names"):
    for f in sorted((WORK/"results2_smoke"/model).glob("s2_*/*.npz")):
        d = np.load(f)
        for key in ("logits","scores","random","pooled","last"):
            assert np.isfinite(d[key].astype(np.float64)).all(),f"{f} has a non finite {key}"
        assert d["scores"].shape[1:]==d["random"].shape[1:2]+(6,4),f"{f} has the wrong score shape"
        assert (d["ink_tokens"]>=0).all() and (d["tokens"]>0).all(),f"{f} has an empty region"
    hits = []
    for f in sorted((WORK/"results2_smoke"/model).glob("s2_named_*/*.npz")):
        d = np.load(f)
        answer = {json.loads(l)["id"]:"ABCD".index(json.loads(l)["answer"]) for l in (WORK/"stimuli2"/f.parent.name/"items.jsonl").read_text().splitlines()}
        hits += list(d["scores"][:,3,0].argmax(-1)==np.array([answer[i] for i in d["ids"]]))
    # named shapes were read at about one hundred percent at stream three in the first study so far less means a broken pipeline
    named_probe = float(np.mean(hits))
    print(f"smoke {model}: named full view probe at stream 3 is {named_probe:.2f}")
    assert named_probe>=0.8,"the probe fails on named shapes so something in the pipeline is broken"
inter = [np.load(f)["logits"] for f in (WORK/"results2_smoke"/"qwen"/"intervene").glob("*/*.npz")]
assert inter and all(np.isfinite(x).all() for x in inter),"the intervention smoke run is missing or not finite"
spread = np.concatenate(inter).std(1).mean()
print(f"smoke intervention: mean spread of the letter logits over conditions {spread:.4f}")
assert spread>0,"the interventions changed nothing so the hooks did not run"

# 4 the main run one model per process so a crash in one cannot spoil the other
for model in ("qwen","qwen_names"):
    if time.time()<STOP_AT:
        step(f"main {model}",[sys.executable,"-m","nwf.s2run","--model",model,"--stop-at",str(STOP_AT)],base_env)

# 5 the analysis on cpu only when every chunk of both models exists
sets = ["s2_named_fill","s2_named_line","s2_unnamed_fill","s2_unnamed_line","s2_named_line_same","s2_unnamed_line_same"]
missing = [f"{m}/{s}/{c:04d}" for m in ("qwen","qwen_names") for s in sets for c in range(0,300,50) if not (WORK/"results2"/m/s/f"{c:04d}.done.json").exists()]
missing += [f"qwen/intervene/{s}/{c:04d}" for s in sets[:4] for c in range(0,200,50) if not (WORK/"results2"/"qwen"/"intervene"/s/f"{c:04d}.done.json").exists()]
if missing:
    print(f"not finished, so the analysis is skipped: {missing}")
    print("add this version's output as an input of the notebook and run again; finished chunks are copied over and skipped")
else:
    analysis_env = dict(base_env)
    for model in ("qwen","qwen_names"):
        for kind in ("pairs","letters"):
            step(f"layer probe {kind} {model}",[sys.executable,"-m","nwf.analysis.layer_probe","--results",str(WORK/"results2"/model),"--stimuli",str(WORK/"stimuli2"),"--sets",",".join(sets),"--kind",kind,"--label",f"s2_{model}"],analysis_env)
    step("report",[sys.executable,"-m","nwf.analysis.s2report"],analysis_env)
    subprocess.run(["cp","-r",str(CODE/"analysis"),str(WORK/"analysis")],check=True)
    print("the results are in /kaggle/working/analysis: study2_report.json, layer_probe_*.json and the study2 figures")
print(f"finished after {round((time.time()-START)/3600,2)} h")
