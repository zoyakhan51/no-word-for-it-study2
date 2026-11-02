import argparse
import hashlib
import json
import shutil

import numpy as np

from nwf import config
from nwf import s2config
from nwf import shapes
from nwf.paths import STIMULI2
from nwf.stimuli import complexity
from nwf.stimuli import lookalike
from nwf.stimuli import place
from nwf.stimuli import render
from nwf.stimuli import sha
from nwf.stimuli import signature


def pool_for(spec,rng):
    """Draw the shapes of one study two set"""
    if spec["kind"]=="named":
        return {n:shapes.draw(parts,"fill" if spec["filled"] else "outline") for n,parts in shapes.NAMED.items()}
    if spec["kind"]=="blob":
        return {f"blob_{i:02d}":shapes.draw([(shapes.blob(rng),"add")],"fill") for i in range(config.POOL)}
    return {f"squig{spec['anchors']}_{i:02d}":shapes.draw([(shapes.squiggle(rng,spec["anchors"]),"add")],"open") for i in range(config.POOL)}


def build(items,names=None):
    """Make the study two sets with the same layout code as the first study and write a manifest of hashes"""
    STIMULI2.mkdir(parents=True,exist_ok=True)
    manifest = {"items_per_set":items,"sets":{}}
    for name in names or s2config.SETS:
        spec = s2config.SETS[name]
        folder = STIMULI2/name
        if folder.exists():
            shutil.rmtree(folder)
        (folder/"shapes").mkdir(parents=True)
        pool = pool_for(spec,np.random.default_rng(spec["seed"]))
        for n,image in pool.items():
            image.save(folder/"shapes"/f"{n}.png")
        signatures = {n:signature(image) for n,image in pool.items()}
        scores = {n:complexity(image) for n,image in pool.items()}
        ids = list(pool)
        # big pools are split in two halves so a probe can be tested on shapes it never saw
        halves = [ids[0::2],ids[1::2]] if len(ids)>=config.POOL else [ids,ids]
        order = np.random.default_rng([spec["seed"],10**6]).permutation(np.tile(np.arange(4),items//4+1)[:items])
        rows = []
        for i in range(items):
            r = np.random.default_rng([spec["seed"],i])
            item = f"{name}_{i:04d}"
            chosen = [str(c) for c in r.choice(halves[i%2],4,replace=False)]
            ref = int(order[i])
            labels = [None]*4
            labels[ref] = "REF"
            images = [pool[c] for c in chosen]
            dark,light = r.integers(0,128,3),r.integers(180,255,3)
            ref_boxes = render(images,place(r),labels,folder/f"{item}_ref.jpg",dark,light)
            # drawing new colours for the second picture unless both pictures share one colour pair
            dark2,light2 = (dark,light) if spec["same_colour"] else (r.integers(0,128,3),r.integers(180,255,3))
            tgt_boxes = render(images,place(r),config.LETTERS,folder/f"{item}_tgt.jpg",dark2,light2)
            shape_ref = chosen[ref]
            rows.append({
                "id":item,"set":name,"shapes":chosen,"ref_index":ref,"answer":config.LETTERS[ref],
                "ref_image":f"{name}/{item}_ref.jpg","tgt_image":f"{name}/{item}_tgt.jpg",
                "ref_boxes":ref_boxes,"tgt_boxes":tgt_boxes,"named":spec["named"],"filled":spec["filled"],
                "same_colour":spec["same_colour"],"half":i%2,
                "complexity":[scores[s] for s in chosen],
                "lookalike":lookalike(signatures[shape_ref],[signatures[s] for s in chosen if s!=shape_ref]),
                "calibration":i<s2config.CALIBRATION,
            })
        with open(folder/"items.jsonl","w",encoding="utf-8",newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row)+"\n")
        files = sorted(p for p in folder.rglob("*") if p.is_file())
        listing = "".join(f"{p.relative_to(STIMULI2).as_posix()} {sha(p)}\n" for p in files)
        letters = [row["answer"] for row in rows]
        manifest["sets"][name] = {
            "items":len(rows),"shapes":len(pool),"files":len(files),
            "letters":{l:letters.count(l) for l in config.LETTERS},
            "items_sha256":sha(folder/"items.jsonl"),
            "all_files_sha256":hashlib.sha256(listing.encode()).hexdigest(),
        }
        print(f"{name}: {len(rows)} items from {len(pool)} shapes, letters {manifest['sets'][name]['letters']}")
    (STIMULI2/"manifest.json").write_text(json.dumps(manifest,indent=1),encoding="utf-8",newline="\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build the study two stimulus sets")
    parser.add_argument("--items",type=int,default=s2config.ITEMS,help="items per set")
    parser.add_argument("--sets",help="comma separated sets to build instead of all of them")
    args = parser.parse_args()
    build(args.items,args.sets.split(",") if args.sets else None)
