import argparse
import json
import time
from functools import partial

import numpy as np
import torch
from PIL import Image

from nwf import config
from nwf import maxsim
from nwf import models
from nwf import s2config
from nwf import tokens
from nwf.paths import RESULTS2
from nwf.paths import STIMULI2
from nwf.run_items import DEVICE
from nwf.run_items import accumulate
from nwf.run_items import capture
from nwf.run_items import encode
from nwf.run_items import forward
from nwf.run_items import letter_ids
from nwf.run_items import locate
from nwf.run_items import save
from nwf.run_items import top_directions

CHUNK = 50


def rows_of(set_name):
    """Read the items of one study two set"""
    return [json.loads(line) for line in (STIMULI2/set_name/"items.jsonl").read_text(encoding="utf-8").splitlines()]


def images_of(row):
    """Open the two pictures of an item"""
    return [Image.open(STIMULI2/row[key]).convert("RGB") for key in ("ref_image","tgt_image")]


def image_spans(ids,start_id,end_id):
    """Return the token positions of each picture between its start and end marker"""
    ids = ids.tolist()
    starts = [i for i,t in enumerate(ids) if t==start_id]
    return [list(range(s+1,ids.index(end_id,s+1))) for s in starts]


def ink_shares(enc,row,images,start_id,end_id):
    """Return for each of the five regions the share of ink inside the shape box for every token in the order the region lists them"""
    grid = enc["image_grid_thw"]
    boxes = [(0,row["ref_boxes"][row["ref_index"]],row["shapes"][row["ref_index"]])]
    boxes += [(1,box,shape) for box,shape in zip(row["tgt_boxes"],row["shapes"])]
    shares = []
    for image_index,box,shape in boxes:
        size = images[image_index].size
        height,width = int(grid[image_index][1])*16,int(grid[image_index][2])*16
        sx,sy = width/size[0],height/size[1]
        # the shape picture is stretched over its box when the item is drawn so its ink mask is stretched the same way
        bx0,by0,bx1,by1 = box[0][0],box[0][1],box[1][0],box[1][1]
        mask = Image.open(STIMULI2/row["set"]/"shapes"/f"{shape}.png").convert("L").resize((max(1,bx1-bx0),max(1,by1-by0)),Image.BILINEAR)
        ink = np.array(mask)<128
        x0,y0,x1,y1 = int(bx0*sx),int(by0*sy),int(bx1*sx),int(by1*sy)
        rows,cols = height//32,width//32
        region = []
        # walking the token blocks in the same order as the token positions are listed
        for gy in range(max(0,y0//32),min(rows,(y1+31)//32)):
            for gx in range(max(0,x0//32),min(cols,(x1+31)//32)):
                if gx*32<x1 and (gx+1)*32>x0 and gy*32<y1 and (gy+1)*32>y0:
                    px0,px1 = max(bx0,int(gx*32/sx)),min(bx1,int((gx+1)*32/sx))
                    py0,py1 = max(by0,int(gy*32/sy)),min(by1,int((gy+1)*32/sy))
                    part = ink[max(0,py0-by0):max(0,py1-by0),max(0,px0-bx0):max(0,px1-bx0)]
                    region.append(float(part.mean()) if part.size else 0.0)
        shares.append(region)
    return shares


def keep_ink(x,share):
    """Keep the tokens with enough ink and the inkiest one when none has enough"""
    chosen = [i for i,v in enumerate(share) if v>=s2config.INK_SHARE]
    return x[chosen] if chosen else x[[int(np.argmax(share))]]


def project_out(x,u):
    """Remove the part of each row that lies in the span of the orthonormal columns of u"""
    return x-(x@u)@u.T


def views(ref,targets,ref_mean,tgt_mean,ref_ink,tgt_ink,basis,randoms):
    """Return the MaxSim scores of every named view and of every random removal for one stream index"""
    loud = {k:basis[:,:k] for k in s2config.LOUD_KS}
    named = [
        maxsim.maxsim(ref,targets),
        maxsim.maxsim(keep_ink(ref,ref_ink),[keep_ink(t,s) for t,s in zip(targets,tgt_ink)]),
        maxsim.maxsim(project_out(ref,loud[16]),[project_out(t,loud[16]) for t in targets]),
        maxsim.maxsim(project_out(ref,loud[64]),[project_out(t,loud[64]) for t in targets]),
        maxsim.maxsim(ref-ref_mean,[t-tgt_mean for t in targets]),
        maxsim.maxsim(project_out(keep_ink(ref,ref_ink),loud[64]),[project_out(keep_ink(t,s),loud[64]) for t,s in zip(targets,tgt_ink)]),
    ]
    random = [maxsim.maxsim(project_out(ref,r),[project_out(t,r) for t in targets]) for r in randoms]
    return torch.stack(named),torch.stack(random)


def random_bases(d):
    """Draw the seeded random subspaces used as the size matched control"""
    out = []
    for i in range(s2config.RANDOM_DRAWS):
        gauss = np.random.default_rng([config.RANDOM_SEED,500+i]).standard_normal((d,s2config.RANDOM_K))
        out.append(torch.from_numpy(np.linalg.qr(gauss)[0]).float().to(DEVICE))
    return out


def fit_loud(model,processor,store,name,image_id,n_items,sets):
    """Fit the top principal directions of the image tokens at every stream index and save them"""
    out = RESULTS2/name/"bases.npz"
    if out.with_suffix(".done.json").exists():
        return
    started = time.time()
    k = max(s2config.LOUD_KS)
    sums = {"shift":None,"count":0}
    energy = None
    for set_name in sets:
        for row in rows_of(set_name)[:n_items]:
            enc = encode(processor,images_of(row),config.PROMPT,config.PREFILL)
            _,states = forward(model,store,enc,tokens.image_positions(enc["input_ids"][0],image_id))
            x = torch.stack(states)
            energy = (x**2).sum((1,2)).double() if energy is None else energy+(x**2).sum((1,2)).double()
            accumulate(sums,x)
    streams = sums["gram"].shape[0]
    fits = [top_directions(sums["gram"][s],sums["total"][s],sums["count"],k) for s in range(streams)]
    image = torch.stack([f[0] for f in fits])
    # the share of the raw image token energy that lies in the top directions shows how loud they are
    caught = torch.zeros(streams,len(s2config.LOUD_KS),dtype=torch.float64,device=DEVICE)
    for set_name in sets:
        for row in rows_of(set_name)[:n_items]:
            enc = encode(processor,images_of(row),config.PROMPT,config.PREFILL)
            _,states = forward(model,store,enc,tokens.image_positions(enc["input_ids"][0],image_id))
            x = torch.stack(states)
            for j,kk in enumerate(s2config.LOUD_KS):
                caught[:,j] += ((x@image[:,:,:kk])**2).sum((1,2)).double()
    arrays = {"image":image.cpu().numpy(),"image_values":torch.stack([f[1] for f in fits]).cpu().numpy(),"loud_share":(caught/energy[:,None]).float().cpu().numpy()}
    save(out,arrays,{"model":name,"image_tokens":sums["count"],"items_per_set":n_items,"seconds":round(time.time()-started,1)})
    print(f"{name}: loud directions from {sums['count']} image tokens in {round(time.time()-started,1)} s")


def score(model,processor,store,name,limit,sets,stop_at):
    """Run every item once and save the letters the probe views the pooled regions and the answer position state at every stream"""
    tokenizer = processor.tokenizer
    letters = letter_ids(tokenizer)
    names = config.FAMILIES["qwen"]
    start_id,end_id = tokenizer.convert_tokens_to_ids([names["start"],names["end"]])
    saved = np.load(RESULTS2/name/"bases.npz")
    basis = torch.from_numpy(saved["image"]).float().to(DEVICE)
    randoms = random_bases(basis.shape[1])
    for set_name in sets:
        rows = rows_of(set_name)[:limit] if limit else rows_of(set_name)
        for first in range(0,len(rows),CHUNK):
            out = RESULTS2/name/set_name/f"{first:04d}.npz"
            if out.with_suffix(".done.json").exists():
                continue
            if stop_at and time.time()>stop_at:
                print(f"{name}: deadline reached before {set_name} {first}")
                return
            began = time.time()
            keep = {"logits":[],"scores":[],"random":[],"pooled":[],"last":[],"ink_tokens":[],"tokens":[]}
            for row in rows[first:first+CHUNK]:
                images = images_of(row)
                enc = encode(processor,images,config.PROMPT,config.PREFILL)
                ids = enc["input_ids"][0]
                regions,_ = locate(enc,row,"qwen",tokenizer,images)
                shares = ink_shares(enc,row,images,start_id,end_id)
                assert [len(r) for r in regions]==[len(s) for s in shares],"ink shares do not line up with the region tokens"
                spans = image_spans(ids,start_id,end_id)
                # the stream is kept at the regions then both whole pictures then the last token where the answer is read
                wanted = sum(regions,[])+spans[0]+spans[1]+[len(ids)-1]
                cuts = np.cumsum([len(r) for r in regions]+[len(spans[0]),len(spans[1])])
                logits,states = forward(model,store,enc,wanted)
                scores,randoms_out,pooled,last = [],[],[],[]
                for s,x in enumerate(states):
                    parts = torch.tensor_split(x,cuts.tolist())
                    ref,targets = parts[0],list(parts[1:5])
                    ref_mean,tgt_mean = parts[5].mean(0),parts[6].mean(0)
                    named,random = views(ref,targets,ref_mean,tgt_mean,shares[0],shares[1:],basis[s],randoms)
                    scores.append(named)
                    randoms_out.append(random)
                    pooled.append(torch.stack([p.mean(0) for p in parts[:5]]))
                    last.append(parts[7][0])
                keep["logits"].append(logits[letters].cpu())
                keep["scores"].append(torch.stack(scores).cpu())
                keep["random"].append(torch.stack(randoms_out).cpu())
                keep["pooled"].append(torch.stack(pooled).half().cpu())
                keep["last"].append(torch.stack(last).half().cpu())
                keep["ink_tokens"].append(torch.tensor([sum(v>=s2config.INK_SHARE for v in s) for s in shares]))
                keep["tokens"].append(torch.tensor([len(r) for r in regions]))
            item_ids = [row["id"] for row in rows[first:first+CHUNK]]
            arrays = {"ids":np.array(item_ids)}
            arrays.update({key:torch.stack(value).numpy() for key,value in keep.items()})
            meta = {"model":name,"set":set_name,"first":first,"items":len(item_ids),"views":s2config.VIEWS,"seconds":round(time.time()-began,1)}
            save(out,arrays,meta)
        print(f"{name}: {set_name} scored")


def turn_down(module,args,positions,directions,scale,mode):
    """Turn the image tokens entering a layer down along some directions or shrink them to the same length"""
    x = args[0].clone()
    where = positions.to(x.device)
    part = x[0,where].float()
    u = directions.to(part.device)
    lowered = part-(1-scale)*(part@u)@u.T
    if mode=="uniform":
        # keeping each token's direction and giving it the length the loud turn down would give
        lowered = part*(lowered.norm(dim=-1,keepdim=True)/part.norm(dim=-1,keepdim=True).clamp_min(1e-12))
    x[0,where] = lowered.to(x.dtype)
    return (x,)+args[1:]


def intervene(model,processor,name,limit,stop_at):
    """Score the letters of the test items with the loud directions turned down at one layer at a time and with both controls"""
    tokenizer = processor.tokenizer
    letters = letter_ids(tokenizer)
    image_id = tokenizer.convert_tokens_to_ids(config.FAMILIES["qwen"]["image"])
    saved = np.load(RESULTS2/name/"bases.npz")
    basis = torch.from_numpy(saved["image"]).float()
    k = s2config.RANDOM_K
    gauss = np.random.default_rng([config.RANDOM_SEED,900]).standard_normal((basis.shape[1],k))
    random = torch.from_numpy(np.linalg.qr(gauss)[0]).float()
    layers = model.model.language_model.layers
    plan = [(layer,condition) for layer in s2config.INTERVENE_LAYERS for condition in s2config.INTERVENE_CONDITIONS]
    for set_name in s2config.INTERVENE_SETS:
        rows = [r for r in rows_of(set_name) if not r["calibration"]]
        rows = rows[:limit] if limit else rows
        for first in range(0,len(rows),CHUNK):
            out = RESULTS2/name/"intervene"/set_name/f"{first:04d}.npz"
            if out.with_suffix(".done.json").exists():
                continue
            if stop_at and time.time()>stop_at:
                print(f"{name}: deadline reached before intervention {set_name} {first}")
                return
            began = time.time()
            logits = []
            for row in rows[first:first+CHUNK]:
                enc = encode(processor,images_of(row),config.PROMPT,config.PREFILL).to(DEVICE)
                positions = torch.tensor(tokens.image_positions(enc["input_ids"][0],image_id))
                per = []
                for layer,condition in plan:
                    directions = random if condition=="random64" else basis[layer][:,:k]
                    mode = "uniform" if condition=="uniform" else "directions"
                    # the turn down goes in front of any capture hook so a later read would see what the layer really reads
                    handle = layers[layer].register_forward_pre_hook(partial(turn_down,positions=positions,directions=directions,scale=s2config.INTERVENE_SCALE,mode=mode),prepend=True)
                    with torch.no_grad():
                        per.append(model(**enc,logits_to_keep=1).logits[0,-1,letters].float().cpu())
                    handle.remove()
                logits.append(torch.stack(per))
            item_ids = [row["id"] for row in rows[first:first+CHUNK]]
            meta = {"model":name,"set":set_name,"first":first,"items":len(item_ids),"plan":[f"{l}:{c}" for l,c in plan],"scale":s2config.INTERVENE_SCALE,"seconds":round(time.time()-began,1)}
            save(out,{"ids":np.array(item_ids),"logits":torch.stack(logits).numpy()},meta)
        print(f"{name}: intervention on {set_name} scored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run one model over the study two sets")
    parser.add_argument("--model",required=True,choices=list(s2config.MODELS))
    parser.add_argument("--items",type=int,help="only the first items of each set for the smoke run")
    parser.add_argument("--stop-at",type=float,help="unix time after which no new chunk starts")
    args = parser.parse_args()
    model,processor = models.load(args.model,"fast")
    store = capture(model)
    sets = list(s2config.SETS)
    image_id = processor.tokenizer.convert_tokens_to_ids(config.FAMILIES["qwen"]["image"])
    fit_loud(model,processor,store,args.model,image_id,2 if args.items else s2config.BASIS_ITEMS,sets)
    score(model,processor,store,args.model,args.items,sets,args.stop_at)
    if s2config.MODELS[args.model]:
        intervene(model,processor,args.model,args.items//2 if args.items else None,args.stop_at)
