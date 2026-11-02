import json

import numpy as np
import pytest
import torch
from PIL import Image

from nwf import config
from nwf import s2config
from nwf import s2run
from nwf import s2stimuli
from nwf import tokens
from nwf.analysis import layer_probe
from nwf.analysis import s2report


def small_sets(monkeypatch,tmp_path,items):
    """Build every study two set with a few items into a temporary folder"""
    monkeypatch.setattr(s2stimuli,"STIMULI2",tmp_path/"stimuli2")
    monkeypatch.setattr(s2run,"STIMULI2",tmp_path/"stimuli2")
    monkeypatch.setattr(s2config,"CALIBRATION",2)
    return s2stimuli.build(items)


def test_stimuli_are_balanced_and_share_colours_only_where_asked(monkeypatch,tmp_path):
    manifest = small_sets(monkeypatch,tmp_path,8)
    assert set(manifest["sets"])==set(s2config.SETS)
    for name,spec in s2config.SETS.items():
        assert manifest["sets"][name]["letters"]=={l:2 for l in config.LETTERS}
        rows = [json.loads(l) for l in (tmp_path/"stimuli2"/name/"items.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [r["half"] for r in rows]==[i%2 for i in range(8)]
        assert [r["calibration"] for r in rows]==[i<2 for i in range(8)]
        same = []
        for r in rows:
            # the top left pixel is background in both pictures because shapes are kept away from the edges
            a = np.array(Image.open(tmp_path/"stimuli2"/r["ref_image"]).convert("RGB"))[2,2].astype(int)
            b = np.array(Image.open(tmp_path/"stimuli2"/r["tgt_image"]).convert("RGB"))[2,2].astype(int)
            same.append(np.abs(a-b).max()<=6)
        assert all(same) if spec["same_colour"] else not any(same)


def test_building_twice_gives_identical_files(monkeypatch,tmp_path):
    first = small_sets(monkeypatch,tmp_path,4)
    second = small_sets(monkeypatch,tmp_path,4)
    assert first==second


def test_ink_shares_line_up_with_the_region_tokens(monkeypatch,tmp_path):
    small_sets(monkeypatch,tmp_path,4)
    start_id,end_id = 900,901
    for name in ("s2_named_fill","s2_unnamed_line"):
        row = json.loads((tmp_path/"stimuli2"/name/"items.jsonl").read_text(encoding="utf-8").splitlines()[0])
        images = [Image.open(tmp_path/"stimuli2"/row[k]).convert("RGB") for k in ("ref_image","tgt_image")]
        # the processor turns an 800 by 600 picture into a 38 by 50 grid of 16 pixel patches
        grid = torch.tensor([[1,38,50],[1,38,50]])
        count = 19*25
        ids = torch.tensor([start_id]+[5]*count+[end_id]+[start_id]+[5]*count+[end_id]+[7,8])
        enc = {"image_grid_thw":grid,"input_ids":ids[None]}
        shares = s2run.ink_shares(enc,row,images,start_id,end_id)
        regions = [tokens.qwen_tokens(ids,grid[0],images[0].size,row["ref_boxes"][row["ref_index"]],0,start_id,end_id)]
        regions += [tokens.qwen_tokens(ids,grid[1],images[1].size,b,1,start_id,end_id) for b in row["tgt_boxes"]]
        assert [len(r) for r in regions]==[len(s) for s in shares]
        assert all(0<=v<=1 for s in shares for v in s)
        if name=="s2_named_fill":
            filled = np.mean([np.mean(s) for s in shares])
        else:
            line = np.mean([np.mean(s) for s in shares])
    # a filled shape covers far more of its tokens than a thin squiggle does
    assert filled>2*line
    spans = s2run.image_spans(ids,start_id,end_id)
    assert [len(s) for s in spans]==[count,count] and spans[0][0]==1


def test_views_find_a_copy_and_removals_do_what_they_say():
    torch.manual_seed(0)
    d = 96
    basis = torch.linalg.qr(torch.randn(d,64))[0]
    ref = torch.randn(5,d)
    targets = [torch.randn(5,d) for _ in range(4)]
    targets[2] = ref.clone()
    randoms = [torch.linalg.qr(torch.randn(d,64))[0] for _ in range(3)]
    named,random = s2run.views(ref,targets,ref.mean(0),torch.stack(targets).mean((0,1)),[0.5]*5,[[0.5]*5]*4,basis,randoms)
    assert named.shape==(len(s2config.VIEWS),4) and random.shape==(3,4)
    assert (named.argmax(-1)==2).all() and (random.argmax(-1)==2).all()
    x = torch.randn(7,d)
    u = basis[:,:16]
    assert torch.allclose(s2run.project_out(x,u)@u,torch.zeros(7,16),atol=1e-5)
    assert torch.equal(s2run.keep_ink(x,[0.0,0.2,0.05,0.3,0,0,0]),x[[1,3]])
    assert torch.equal(s2run.keep_ink(x,[0.0,0.02,0.05,0.01,0,0,0]),x[[2]])


def test_turn_down_halves_the_loud_part_and_uniform_only_matches_its_length():
    torch.manual_seed(1)
    d = 40
    u = torch.linalg.qr(torch.randn(d,8))[0]
    x = torch.randn(1,6,d)
    positions = torch.tensor([1,2,4])
    (loud,) = s2run.turn_down(None,(x,),positions,u,0.5,"directions")
    (same,) = s2run.turn_down(None,(x,),positions,u,0.5,"uniform")
    # tokens outside the image are untouched
    assert torch.equal(loud[0,[0,3,5]],x[0,[0,3,5]]) and torch.equal(same[0,[0,3,5]],x[0,[0,3,5]])
    assert torch.allclose(loud[0,positions]@u,0.5*(x[0,positions]@u),atol=1e-5)
    outside = loud[0,positions]-(loud[0,positions]@u)@u.T
    assert torch.allclose(outside,x[0,positions]-(x[0,positions]@u)@u.T,atol=1e-5)
    assert torch.allclose(same[0,positions].norm(dim=-1),loud[0,positions].norm(dim=-1),atol=1e-4)
    cos = torch.nn.functional.cosine_similarity(same[0,positions],x[0,positions],dim=-1)
    assert torch.allclose(cos,torch.ones(3),atol=1e-5)


def fake_pooled(path,stim,name,rng,streams,mode):
    """Write fake items and pooled vectors where the shape signal is kept or drowned or lost in the middle streams"""
    n,d = 120,48
    pool = [f"{name}_{j}" for j in range(20)]
    identity = {s:rng.standard_normal(d) for s in pool}
    rows,pooled = [],[]
    for i in range(n):
        half = pool[i%2::2]
        chosen = list(rng.choice(half,4,replace=False))
        ref = int(rng.integers(4))
        rows.append({"id":f"{name}_{i:04d}","answer":"ABCD"[ref],"shapes":chosen,"ref_index":ref,"calibration":i<20,"half":i%2})
        item = np.zeros((streams,5,d))
        for s in range(streams):
            middle = streams//3<=s<2*streams//3
            signal = 0.0 if (mode=="lost" and middle) else 1.0
            vecs = [identity[chosen[ref]]]+[identity[c] for c in chosen]
            item[s] = [signal*v+0.3*rng.standard_normal(d) for v in vecs]
            if mode=="drowned" and middle:
                # large noise along a few fixed directions that differs from token to token like the loud directions
                item[s][:,:6] += 30*rng.standard_normal((5,6))
        pooled.append(item)
    (stim/name).mkdir(parents=True)
    (stim/name/"items.jsonl").write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
    (path/name).mkdir(parents=True)
    np.savez(path/name/"0000.npz",ids=np.array([r["id"] for r in rows]),pooled=np.array(pooled).astype(np.float16))


def test_layer_probe_tells_lost_from_drowned(monkeypatch,tmp_path):
    rng = np.random.default_rng(3)
    monkeypatch.setattr(layer_probe,"ANALYSIS",tmp_path/"analysis")
    monkeypatch.setattr(s2config,"PROBE_DIMS",24)
    for name,mode in (("kept","kept"),("drowned","drowned"),("lost","lost")):
        fake_pooled(tmp_path/"res",tmp_path/"stim",name,rng,9,mode)
    out = layer_probe.run(tmp_path/"res",tmp_path/"stim",["kept","drowned","lost"],["kept","lost"],0,"pooled","pairs","test")
    s = out["sets"]
    # the trained probe reads the shape through a shared loud part but not when the shape is gone
    assert s["drowned"]["probe"]["drop_early_minus_middle"]["value"]<0.1
    assert s["lost"]["probe"]["drop_early_minus_middle"]["value"]>0.4
    # plain cosine is fooled by the shared loud part
    assert s["drowned"]["cosine"]["drop_early_minus_middle"]["value"]>0.3
    assert s["kept"]["probe"]["early"]["value"]>0.9
    assert out["drop_difference"]["probe"]["low"]>0


def test_letter_probe_reads_a_planted_answer_and_its_control_stays_near_chance(monkeypatch,tmp_path):
    rng = np.random.default_rng(4)
    monkeypatch.setattr(layer_probe,"ANALYSIS",tmp_path/"analysis")
    monkeypatch.setattr(s2config,"PROBE_DIMS",16)
    n,streams,d = 200,6,32
    letters = rng.normal(size=(4,d))
    rows,last = [],[]
    for i in range(n):
        ref = int(rng.integers(4))
        rows.append({"id":f"x_{i:04d}","answer":"ABCD"[ref],"shapes":[f"s{(i+j)%10}" for j in range(4)],"ref_index":ref,"calibration":i<40,"half":i%2})
        # the answer is written into the late streams only
        last.append([(1.0 if s>=3 else 0.0)*letters[ref]+rng.normal(size=d) for s in range(streams)])
    (tmp_path/"stim"/"x").mkdir(parents=True)
    (tmp_path/"stim"/"x"/"items.jsonl").write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
    (tmp_path/"res"/"x").mkdir(parents=True)
    np.savez(tmp_path/"res"/"x"/"0000.npz",ids=np.array([r["id"] for r in rows]),last=np.array(last).astype(np.float16))
    out = layer_probe.run(tmp_path/"res",tmp_path/"stim",["x"],["x","x"],0,"last","letters","test")
    curve = [c["value"] for c in out["sets"]["x"]["probe"]["by_stream"]]
    assert max(curve[:3])<0.4 and min(curve[3:])>0.8
    control = [c["value"] for c in out["sets"]["x"]["control"]["by_stream"]]
    assert max(control)<0.4


def test_holm_and_log_odds_match_hand_values():
    assert s2report.holm([0.01,0.04,0.03,0.5])==pytest.approx([0.04,0.09,0.09,0.5])
    logits = np.log(np.array([[0.4,0.2,0.2,0.2]]))
    assert s2report.log_odds(logits,np.array([0]))[0]==pytest.approx(np.log(0.4/0.6))


def fake_run(root,model,stim,rng,names_taught):
    """Write fake study two results with planted effects for one model"""
    n,streams,first_test = 60,12,10
    v = len(s2config.VIEWS)
    for s,spec in s2config.SETS.items():
        rows = []
        for i in range(n):
            ref = i%4
            rows.append({"id":f"{s}_{i:04d}","answer":"ABCD"[ref],"shapes":[f"{s}_{(i+j)%8}" for j in range(4)],"ref_index":ref,"calibration":i<first_test})
        (stim/s).mkdir(parents=True,exist_ok=True)
        (stim/s/"items.jsonl").write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
        answer = np.array([i%4 for i in range(n)])
        hit = np.ones((n,streams,v),dtype=bool)
        rand_hit = np.ones((n,streams,s2config.RANDOM_DRAWS),dtype=bool)
        unnamed_line = s in ("s2_unnamed_line","s2_unnamed_line_same")
        if unnamed_line and not names_taught:
            middle = slice(6,9)
            hit[:,middle,:] = rng.random((n,3,v))<0.3
            hit[:,middle,s2config.VIEWS.index("loud64_out")] = True
            rand_hit[:,middle,:] = rng.random((n,3,s2config.RANDOM_DRAWS))<0.3
        def as_scores(h,shape):
            out = np.zeros(shape+(4,),dtype=np.float32)
            pick = np.where(h,answer.reshape((n,)+(1,)*(len(shape)-1)),(answer.reshape((n,)+(1,)*(len(shape)-1))+1)%4)
            np.put_along_axis(out,pick[...,None],1.0,-1)
            return out
        share = 0.7 if "unnamed" not in s else (0.8 if s.endswith("same") else 0.3)
        # a fixed share of right answers spread evenly over the shapes keeps the planted gap free of sampling noise
        right = (np.arange(n)*7)%10<share*10
        logits = as_scores(right,(n,))
        folder = root/model/s
        folder.mkdir(parents=True,exist_ok=True)
        np.savez(folder/"0000.npz",ids=np.array([r["id"] for r in rows]),scores=as_scores(hit,(n,streams,v)),random=as_scores(rand_hit,(n,streams,s2config.RANDOM_DRAWS)),logits=logits)
        if s in s2config.INTERVENE_SETS and s2config.MODELS[model]:
            plan = [(l,c) for l in s2config.INTERVENE_LAYERS for c in s2config.INTERVENE_CONDITIONS]
            test = [i for i in range(n) if i>=first_test]
            after = np.repeat(logits[test][:,None],len(plan),1).copy()
            if s=="s2_unnamed_line":
                # turning the loud directions down at layer 9 adds two to the right letter
                j = plan.index((9,"loud64"))
                after[np.arange(len(test)),j,answer[test]] += 2.0
            out = root/model/"intervene"/s
            out.mkdir(parents=True,exist_ok=True)
            np.savez(out/"0000.npz",ids=np.array([rows[i]["id"] for i in test]),logits=after)


def test_report_applies_the_rules_to_planted_effects(monkeypatch,tmp_path):
    rng = np.random.default_rng(5)
    for module in (s2report,):
        monkeypatch.setattr(module,"RESULTS2",tmp_path/"results2")
        monkeypatch.setattr(module,"STIMULI2",tmp_path/"stim")
        monkeypatch.setattr(module,"ANALYSIS",tmp_path/"analysis")
    fake_run(tmp_path/"results2","qwen",tmp_path/"stim",rng,False)
    fake_run(tmp_path/"results2","qwen_names",tmp_path/"stim",rng,True)
    out = s2report.report(["qwen","qwen_names"])
    rules = out["rules"]
    assert rules["H1 unnamed information fades more than named"]=="supported"
    assert rules["H3 loud directions hide unnamed information"]=="supported"
    assert rules["H6 turning loud directions down helps unnamed answers"]=="supported at layers [9]"
    assert rules["H7 name teaching keeps unnamed information alive"]=="supported"
    assert rules["H4 background colour drives the unnamed failure"]=="supported"
    delta = out["models"]["qwen"]["intervention"]["s2_unnamed_line"]
    assert delta["9:loud64"]["value"]>1.0 and abs(delta["3:loud64"]["value"])<1e-9
    assert (tmp_path/"analysis"/"study2"/"qwen_full_by_stream.png").exists()
