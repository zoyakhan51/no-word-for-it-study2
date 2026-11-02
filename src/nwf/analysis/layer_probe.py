import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from nwf import s2config
from nwf.analysis.behaviour import interval
from nwf.analysis.loudness import bootstrap
from nwf.paths import ANALYSIS


def load(results,stimuli,set_name,key):
    """Return the saved per stream vectors of one set with the answer shape cluster pool half and test flag of every item"""
    items = {}
    for line in (stimuli/set_name/"items.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        items[row["id"]] = row
    loaded = [np.load(f) for f in sorted((results/set_name).glob("*.npz"))]
    ids = np.concatenate([c["ids"] for c in loaded])
    rows = [items[i] for i in ids]
    # the first study items carry no half field but item i drew its shapes from half i mod 2
    half = np.array([r.get("half",int(r["id"].rsplit("_",1)[1])%2) for r in rows])
    test = np.array([not r["calibration"] for r in rows])
    if all(r["calibration"] for r in rows):
        test[:] = True
    return {
        "x":np.concatenate([c[key] for c in loaded]),
        "answer":np.array(["ABCD".index(r["answer"]) for r in rows]),
        "cluster":np.array([r["shapes"][r["ref_index"]] for r in rows]),
        "half":half,"test":test,
    }


def reduce(train,dims):
    """Return the mean and the top principal directions of some training vectors scaled to unit spread"""
    mean = train.mean(0)
    _,_,vectors = np.linalg.svd(train-mean,full_matrices=False)
    keep = vectors[:dims].T
    spread = (train-mean)@keep
    return mean,keep/spread.std(0).clip(1e-6)


def softmax_fit(features,answer,penalty):
    """Fit weights that score each of the four candidates and are trained so the right one gets the highest softmax share"""
    n,k,p = features.shape

    def loss(w):
        """Return the mean negative log share of the right candidate plus the penalty and its gradient"""
        s = features@w
        s = s-s.max(1,keepdims=True)
        logp = s-np.log(np.exp(s).sum(1,keepdims=True))
        grad = np.einsum("nk,nkp->p",np.exp(logp),features)-features[np.arange(n),answer].sum(0)
        return -logp[np.arange(n),answer].mean()+penalty*w@w,grad/n+2*penalty*w

    return minimize(loss,np.zeros(p),jac=True,method="L-BFGS-B").x


def pick(features,answer,rng):
    """Choose the penalty on a split of the items and refit on all of them"""
    inner = rng.permutation(len(features))%2==0
    best,best_acc = s2config.PROBE_L2[0],-1.0
    for penalty in s2config.PROBE_L2:
        w = softmax_fit(features[inner],answer[inner],penalty)
        acc = ((features[~inner]@w).argmax(-1)==answer[~inner]).mean()
        if acc>best_acc:
            best,best_acc = penalty,acc
    return softmax_fit(features,answer,best)


def pair_features(x,mean,proj):
    """Return for every candidate the size of its difference from the reference along each reduced direction and on average"""
    z = (x-mean)@proj
    diff = np.abs(z[:,:1]-z[:,1:])
    # the mean difference lets a strongly penalised probe fall back to the plain untrained distance
    return np.concatenate([diff,diff.mean(-1,keepdims=True)],-1)


def letter_features(x,mean,proj):
    """Return the reduced answer position state once per letter slot so one weight block scores each letter"""
    z = (x-mean)@proj
    out = np.zeros((len(z),4,4*z.shape[1]))
    for c in range(4):
        out[:,c,c*z.shape[1]:(c+1)*z.shape[1]] = z
    return out


def probe_set(d,kind,seed):
    """Return per test item and stream whether the trained probe and its comparisons got the answer"""
    x,answer,half,test = d["x"],d["answer"],d["half"],d["test"]
    streams = x.shape[1]
    # for shape pairs any weighting that prefers the closest candidate solves the task so shuffled labels are no control
    # the comparison is the same distance with every reduced direction weighted equally and the plain cosine
    names = ("probe","untrained","cosine") if kind=="pairs" else ("probe","control")
    hits = {name:np.zeros((test.sum(),streams),dtype=bool) for name in names}
    order = np.flatnonzero(test)
    for s in range(streams):
        xs = x[:,s].astype(np.float64)
        for fold in (0,1):
            # training on one pool half and testing on the other so big pools are tested on shapes the probe never saw
            train = half==fold
            held = (half!=fold)&test
            slot = np.searchsorted(order,np.flatnonzero(held))
            rng = np.random.default_rng([seed,s,fold])
            shuffled = rng.integers(0,4,train.sum())
            flat = xs[train].reshape(-1,xs.shape[-1])
            mean,proj = reduce(flat,min(s2config.PROBE_DIMS,flat.shape[0]-1))
            make = pair_features if kind=="pairs" else letter_features
            train_features,held_features = make(xs[train],mean,proj),make(xs[held],mean,proj)
            hits["probe"][slot,s] = (held_features@pick(train_features,answer[train],rng)).argmax(-1)==answer[held]
            if kind=="pairs":
                hits["untrained"][slot,s] = (-held_features[...,-1]).argmax(-1)==answer[held]
            else:
                hits["control"][slot,s] = (held_features@pick(train_features,shuffled,rng)).argmax(-1)==answer[held]
        if kind=="pairs":
            ref,tgt = xs[test][:,0],xs[test][:,1:]
            cos = (tgt@ref[:,:,None])[...,0]/(np.linalg.norm(tgt,axis=-1)*np.linalg.norm(ref,axis=-1)[:,None]).clip(1e-12)
            hits["cosine"][:,s] = cos.argmax(-1)==answer[test]
    return hits


def thirds(streams,first):
    """Split the stream indices after the first one into early middle and late thirds"""
    return [list(part) for part in np.array_split(np.arange(first,streams),3)]


def summarise(hits,cluster,first,rng):
    """Return per stream accuracies with intervals and the early minus middle drop for every probe kind"""
    out = {}
    for name,h in hits.items():
        point,boot = bootstrap(h.astype(float),cluster,rng)
        early,middle,late = thirds(h.shape[1],first)
        out[name] = {
            "by_stream":[interval(p,b) for p,b in zip(point,boot.T)],
            "early":interval(point[early].mean(),boot[:,early].mean(-1)),
            "middle":interval(point[middle].mean(),boot[:,middle].mean(-1)),
            "late":interval(point[late].mean(),boot[:,late].mean(-1)),
            "drop_early_minus_middle":interval(point[early].mean()-point[middle].mean(),boot[:,early].mean(-1)-boot[:,middle].mean(-1)),
        }
        out[name]["_boot_drop"] = boot[:,early].mean(-1)-boot[:,middle].mean(-1)
    return out


def run(results,stimuli,sets,pair,first,key,kind,label):
    """Probe every set and compare the drop of the pair and write the result"""
    rng = np.random.default_rng(0)
    table = {}
    for set_name in sets:
        d = load(results,stimuli,set_name,key)
        hits = probe_set(d,kind,0)
        table[set_name] = summarise(hits,d["cluster"][d["test"]],first,rng)
        table[set_name]["items"] = int(d["test"].sum())
        p = table[set_name]["probe"]
        print(f"{set_name:22s} n={table[set_name]['items']:4d} probe early {p['early']['value']:.3f} middle {p['middle']['value']:.3f} late {p['late']['value']:.3f} drop {p['drop_early_minus_middle']['value']:+.3f} [{p['drop_early_minus_middle']['low']:+.3f},{p['drop_early_minus_middle']['high']:+.3f}] second {list(table[set_name])[1]} early {table[set_name][list(table[set_name])[1]]['early']['value']:.3f}")
    diffs = {}
    if pair[0] in table and pair[1] in table:
        named,unnamed = table[pair[0]],table[pair[1]]
        for name in named:
            if name.startswith("_") or name=="items":
                continue
            # positive when the unnamed set loses more between the early and the middle third
            point = unnamed[name]["drop_early_minus_middle"]["value"]-named[name]["drop_early_minus_middle"]["value"]
            diffs[name] = interval(point,unnamed[name]["_boot_drop"]-named[name]["_boot_drop"])
            print(f"{pair[1]} minus {pair[0]} drop ({name}): {diffs[name]['value']:+.3f} [{diffs[name]['low']:+.3f},{diffs[name]['high']:+.3f}]")
    for row in table.values():
        for name in list(row):
            if isinstance(row[name],dict):
                row[name].pop("_boot_drop",None)
    # the first study data suggested the question so only the fresh study two items count as a test
    out = {"status":"study2" if label.startswith("s2") else "exploratory","kind":kind,"key":key,"first_stream":first,"pair":list(pair),"sets":table,"drop_difference":diffs}
    ANALYSIS.mkdir(exist_ok=True)
    path = ANALYSIS/f"layer_probe_{kind}_{label}.json"
    path.write_text(json.dumps(out,indent=1),encoding="utf-8")
    print(f"saved {path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trained probes at every stream index on the saved region vectors or answer position states")
    parser.add_argument("--results",required=True,help="folder that holds one subfolder of chunks per set for one model")
    parser.add_argument("--stimuli",required=True,help="folder that holds the items of those sets")
    parser.add_argument("--sets",required=True,help="comma separated sets")
    parser.add_argument("--pair",default=",".join(s2config.MAIN_PAIR),help="named set and unnamed set whose drops are compared")
    parser.add_argument("--first",type=int,default=s2config.FIRST_STREAM,help="first stream index after the extra vision features")
    parser.add_argument("--kind",choices=["pairs","letters"],default="pairs",help="same shape probe on region vectors or letter probe on the answer position")
    parser.add_argument("--label",required=True,help="name for the output file for example old_qwen or s2_qwen")
    args = parser.parse_args()
    key = "pooled" if args.kind=="pairs" else "last"
    run(Path(args.results),Path(args.stimuli),args.sets.split(","),args.pair.split(","),args.first,key,args.kind,args.label)
