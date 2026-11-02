import argparse
import json

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from nwf import s2config
from nwf.analysis.behaviour import interval
from nwf.analysis.loudness import bootstrap
from nwf.paths import ANALYSIS
from nwf.paths import RESULTS2
from nwf.paths import STIMULI2

COLOURS = {"s2_named_fill":"steelblue","s2_named_line":"seagreen","s2_unnamed_fill":"darkorange","s2_unnamed_line":"firebrick","s2_named_line_same":"darkseagreen","s2_unnamed_line_same":"salmon"}


def items_of(set_name):
    """Return the items of one study two set by id"""
    rows = [json.loads(line) for line in (STIMULI2/set_name/"items.jsonl").read_text(encoding="utf-8").splitlines()]
    return {r["id"]:r for r in rows}


def load(model,set_name,folder=""):
    """Return the saved arrays of one set and model joined over chunks with the answer cluster and test flag"""
    loaded = [np.load(f) for f in sorted((RESULTS2/model/folder/set_name).glob("*.npz"))]
    out = {key:np.concatenate([c[key] for c in loaded]) for key in loaded[0].files}
    items = items_of(set_name)
    rows = [items[i] for i in out["ids"]]
    out["answer"] = np.array(["ABCD".index(r["answer"]) for r in rows])
    out["cluster"] = np.array([r["shapes"][r["ref_index"]] for r in rows])
    out["test"] = np.array([not r["calibration"] for r in rows])
    return out


def pvalue(boot):
    """Return a two sided bootstrap p value for a difference being zero"""
    return float(min(1.0,2*min((boot<=0).mean(),(boot>=0).mean())))


def holm(pvalues):
    """Return Holm adjusted p values in the original order"""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank,i in enumerate(order):
        running = max(running,min(1.0,(len(pvalues)-rank)*pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


def thirds(streams):
    """Split the stream indices after the extra vision features into early middle and late thirds"""
    return [list(part) for part in np.array_split(np.arange(s2config.FIRST_STREAM,streams),3)]


def log_odds(logits,answer):
    """Return the log odds of the right letter under a softmax over the four letters"""
    z = logits-logits.max(-1,keepdims=True)
    p = np.exp(z)/np.exp(z).sum(-1,keepdims=True)
    right = np.take_along_axis(p,answer[...,None],-1)[...,0].clip(1e-9,1-1e-9)
    return np.log(right)-np.log(1-right)


def curves(d,rng):
    """Return per test item and stream the hits of every view and the mean hit of the random removals with their summaries"""
    t = d["test"]
    answer = d["answer"][t]
    hits = (d["scores"][t].argmax(-1)==answer[:,None,None]).astype(float)
    random = (d["random"][t].argmax(-1)==answer[:,None,None]).mean(-1)
    stack = np.concatenate([hits,random[...,None]],-1)
    point,boot = bootstrap(stack,d["cluster"][t],rng)
    early,middle,late = thirds(stack.shape[1])
    names = s2config.VIEWS+["random64_out"]
    out = {}
    for v,name in enumerate(names):
        out[name] = {
            "by_stream":point[:,v].round(4).tolist(),
            "early":interval(point[early,v].mean(),boot[:,early,v].mean(-1)),
            "middle":interval(point[middle,v].mean(),boot[:,middle,v].mean(-1)),
            "late":interval(point[late,v].mean(),boot[:,late,v].mean(-1)),
            "drop_early_minus_middle":interval(point[early,v].mean()-point[middle,v].mean(),boot[:,early,v].mean(-1)-boot[:,middle,v].mean(-1)),
        }
    keep = {"drop_full":boot[:,early,0].mean(-1)-boot[:,middle,0].mean(-1),
            "mid_loud_minus_random":boot[:,middle,names.index("loud64_out")].mean(-1)-boot[:,middle,names.index("random64_out")].mean(-1),
            "mid_full":boot[:,middle,0].mean(-1)}
    out["middle_loud64_minus_random"] = interval(point[middle,names.index("loud64_out")].mean()-point[middle,names.index("random64_out")].mean(),keep["mid_loud_minus_random"])
    return out,keep


def behaviour(d,rng):
    """Return the test accuracy of the letters with an interval and its bootstrap draws"""
    t = d["test"]
    hit = (d["logits"][t].argmax(-1)==d["answer"][t]).astype(float)
    point,boot = bootstrap(hit[:,None],d["cluster"][t],rng)
    return interval(point[0],boot[:,0]),boot[:,0]


def plot_curves(model,data,view,path,title):
    """Draw probe accuracy by stream for every set in one view"""
    fig = Figure(figsize=(7,4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    for set_name,c in data.items():
        ax.plot(c[view]["by_stream"],label=set_name.replace("s2_",""),color=COLOURS[set_name])
    ax.axhline(0.25,color="gray",linestyle="--",linewidth=1,label="chance")
    ax.set_xlabel("stream index (layer input)")
    ax.set_ylabel("probe accuracy")
    ax.set_ylim(0.2,1.02)
    ax.set_title(title)
    ax.legend(fontsize=7,loc="lower left")
    fig.tight_layout()
    fig.savefig(path,dpi=150)


def plot_views(c,path,title):
    """Draw probe accuracy by stream for every view of one set"""
    fig = Figure(figsize=(7,4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    styles = {"full":"black","ink":"steelblue","loud16_out":"darkorange","loud64_out":"firebrick","mean_out":"seagreen","ink_loud64_out":"purple","random64_out":"gray"}
    for view,colour in styles.items():
        ax.plot(c[view]["by_stream"],label=view,color=colour)
    ax.axhline(0.25,color="gray",linestyle="--",linewidth=1)
    ax.set_xlabel("stream index (layer input)")
    ax.set_ylabel("probe accuracy")
    ax.set_ylim(0.2,1.02)
    ax.set_title(title)
    ax.legend(fontsize=7,loc="lower left")
    fig.tight_layout()
    fig.savefig(path,dpi=150)


def report(models):
    """Compute every study two result for the given models and check each against the rules fixed before the data"""
    rng = np.random.default_rng(0)
    out = {"models":{},"rules":{}}
    figures = ANALYSIS/"study2"
    figures.mkdir(parents=True,exist_ok=True)
    kept = {}
    for model in models:
        data = {s:load(model,s) for s in s2config.SETS}
        curve,boots,beh = {},{},{}
        for s,d in data.items():
            curve[s],boots[s] = curves(d,rng)
            beh[s],boots[s]["behaviour"] = behaviour(d,rng)
        kept[model] = (data,boots)
        m = {"behaviour":beh,"curves":curve}
        named,unnamed = s2config.MAIN_PAIR
        m["h1_drop_unnamed_minus_named"] = interval(curve[unnamed]["full"]["drop_early_minus_middle"]["value"]-curve[named]["full"]["drop_early_minus_middle"]["value"],boots[unnamed]["drop_full"]-boots[named]["drop_full"])
        m["h3_loud_recovery_unnamed_minus_named"] = interval(curve[unnamed]["middle_loud64_minus_random"]["value"]-curve[named]["middle_loud64_minus_random"]["value"],boots[unnamed]["mid_loud_minus_random"]-boots[named]["mid_loud_minus_random"])
        m["h4_same_colour_minus_random_colour"] = {b:interval(beh[b]["value"]-beh[a]["value"],boots[b]["behaviour"]-boots[a]["behaviour"]) for a,b in s2config.COLOUR_PAIRS}
        if s2config.MODELS[model]:
            m["intervention"],m["h6"] = intervention(model,data,rng)
        out["models"][model] = m
        plot_curves(model,{s:curve[s] for s in s2config.SETS},"full",figures/f"{model}_full_by_stream.png",f"{model}: same-shape probe at every layer")
        plot_views(curve[unnamed],figures/f"{model}_unnamed_line_views.png",f"{model}: unnamed line drawings, every view")
        plot_views(curve[named],figures/f"{model}_named_line_views.png",f"{model}: named line drawings, every view")
    if "qwen" in kept and "qwen_names" in kept:
        out["h7_names_minus_base_middle_full"] = {}
        for s in s2config.SETS:
            base_d,base_b = kept["qwen"]
            names_d,names_b = kept["qwen_names"]
            assert (base_d[s]["ids"]==names_d[s]["ids"]).all(),"the two models scored different items"
            # both models saw the same items so the difference is taken item by item before whole shapes are resampled
            t = base_d[s]["test"]
            early,middle,late = thirds(base_d[s]["scores"].shape[1])
            diff = ((names_d[s]["scores"][t].argmax(-1)==names_d[s]["answer"][t][:,None,None]).astype(float)-(base_d[s]["scores"][t].argmax(-1)==base_d[s]["answer"][t][:,None,None]).astype(float))[:,middle,0].mean(-1)
            p,b = bootstrap(diff[:,None],base_d[s]["cluster"][t],rng)
            out["h7_names_minus_base_middle_full"][s] = interval(p[0],b[:,0])
    out["rules"] = verdicts(out)
    path = ANALYSIS/"study2_report.json"
    path.write_text(json.dumps(out,indent=1),encoding="utf-8")
    for rule,v in out["rules"].items():
        print(f"{rule}: {v}")
    print(f"saved {path} and the figures in {figures}")
    return out


def intervention(model,data,rng):
    """Return the change in the right letter log odds for every layer and condition against no change and the paired contrasts"""
    table,h6 = {},{}
    plan = [(layer,condition) for layer in s2config.INTERVENE_LAYERS for condition in s2config.INTERVENE_CONDITIONS]
    for s in s2config.INTERVENE_SETS:
        d = load(model,s,"intervene")
        base = data[s]
        position = {i:k for k,i in enumerate(base["ids"])}
        rows = [position[i] for i in d["ids"]]
        answer = base["answer"][rows]
        cluster = base["cluster"][rows]
        before = log_odds(base["logits"][rows],answer)
        after = log_odds(d["logits"],answer[:,None])
        delta = after-before[:,None]
        point,boot = bootstrap(delta,cluster,rng)
        table[s] = {f"{layer}:{condition}":interval(point[j],boot[:,j]) for j,(layer,condition) in enumerate(plan)}
        contrasts = {}
        for layer in s2config.INTERVENE_LAYERS:
            j = {c:plan.index((layer,c)) for c in s2config.INTERVENE_CONDITIONS}
            for control in ("uniform","random64"):
                draws = boot[:,j["loud64"]]-boot[:,j[control]]
                contrasts[f"{layer}:loud64_minus_{control}"] = dict(interval(point[j["loud64"]]-point[j[control]],draws),p=pvalue(draws))
        h6[s] = contrasts
    for s in h6:
        keys = list(h6[s])
        adjusted = holm([h6[s][k]["p"] for k in keys])
        for k,a in zip(keys,adjusted):
            h6[s][k]["p_holm"] = a
    return table,h6


def verdicts(out):
    """Apply the rules written in STUDY2.md before the data to the main model"""
    m = out["models"].get("qwen")
    if m is None:
        return {}
    named,unnamed = s2config.MAIN_PAIR
    rules = {}
    h1 = m["h1_drop_unnamed_minus_named"]
    rules["H1 unnamed information fades more than named"] = "supported" if h1["low"]>0 else ("reversed" if h1["high"]<0 else "not supported")
    pairs = ANALYSIS/"layer_probe_pairs_s2_qwen.json"
    if pairs.exists():
        drop = json.loads(pairs.read_text(encoding="utf-8"))["sets"][unnamed]["probe"]["drop_early_minus_middle"]
        rules["H2 the trained probe also loses unnamed information"] = "lost" if drop["value"]>=0.10 and drop["low"]>0 else ("kept but reshaped" if drop["value"]<0.05 and drop["high"]<0.10 else "unclear")
    letters = ANALYSIS/"layer_probe_letters_s2_qwen.json"
    if letters.exists():
        row = json.loads(letters.read_text(encoding="utf-8"))["sets"]
        reach = {s:row[s]["probe"]["late"]["low"]>row[s]["control"]["late"]["high"] for s in (named,unnamed)}
        rules["H5 the answer position carries the shape match in the late layers"] = f"named {reach[named]} unnamed {reach[unnamed]}"
    h3 = m["h3_loud_recovery_unnamed_minus_named"]
    own = m["curves"][unnamed]["middle_loud64_minus_random"]
    rules["H3 loud directions hide unnamed information"] = "supported" if own["low"]>0 and h3["low"]>0 else ("partly" if own["low"]>0 else "not supported")
    h4 = m["h4_same_colour_minus_random_colour"][f"{unnamed}_same"]
    rules["H4 background colour drives the unnamed failure"] = "supported" if h4["low"]>0 and h4["value"]>=0.10 else ("small effect" if h4["low"]>0 else "not supported")
    if "h6" in m:
        hits = [k for k,v in m["h6"][unnamed].items() if v["p_holm"]<0.05 and v["value"]>0]
        layers_both = [l for l in s2config.INTERVENE_LAYERS if f"{l}:loud64_minus_uniform" in hits and f"{l}:loud64_minus_random64" in hits]
        rules["H6 turning loud directions down helps unnamed answers"] = f"supported at layers {layers_both}" if layers_both else "not supported"
    if "h7_names_minus_base_middle_full" in out:
        h7 = out["h7_names_minus_base_middle_full"][unnamed]
        rules["H7 name teaching keeps unnamed information alive"] = "supported" if h7["low"]>0 else "not supported"
    return rules


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="study two report")
    parser.add_argument("--models",default=",".join(s2config.MODELS))
    args = parser.parse_args()
    report(args.models.split(","))
