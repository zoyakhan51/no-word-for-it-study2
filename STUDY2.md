# Study 2: where does the model lose the shape information of unnamed shapes, and why?

Status: **rules fixed before any study 2 data** (2026-11-02). Any later change goes in the log at the end, with the date and the reason, and is reported as a deviation.

## 1. The question in plain words

In study 1, a simple probe could read the shape of a squiggle inside the model, but the model still answered badly. Reading the probe at every layer showed more:

- **Named shapes** (circle, star): the information stays at about 100% through every layer.
- **Unnamed shapes** (squiggles, blobs): the information is strong when it enters the language model, then fades in the middle layers. In Qwen it partly recovers at the end; in Gemma it does not.
- Removing the 64 strongest shared directions of the image tokens ("loud directions") lifts the squiggle probe from 0.74 to 0.98. Removing 64 random directions does nothing.

Study 2 asks four things:

1. **Where** does unnamed shape information fade? (H1)
2. Is it **really lost**, or still there but **covered up** by loud shared directions? (H2, H3)
3. Is part of the problem just **background colour**, which fills most of a thin squiggle's tokens? (H4)
4. Does the information **reach the answer**, and does turning the loud directions down **change the answer**? (H5, H6)

And one link back to names: does the name-taught model **keep** the squiggle information alive? (H7)

Study 1's data suggested these questions, so study 2 tests them on **fresh items** built from new seeds, with the rules below written first.

## 2. What changed from study 1 (fixes from the methodology review)

| Problem in study 1 | Fix in study 2 |
|---|---|
| The loudness idea came from the same data (issue 6) | Fresh items (new seeds 201–206); rules fixed before data |
| E2 read one early layer only (issue 4) | Every layer is the main result, summarised as early, middle and late thirds |
| MaxSim is a crude probe (issue 9) | A trained probe on held-out shapes at every layer, compared with the same distance untrained |
| Thin squiggle tokens are mostly background (issue 5) | An ink-only view, and same-colour item sets |
| Removal broke the model (issue 7) | Gentle edits: halve the loud directions at one layer at a time; score the change in the right letter's log-odds |
| The random control was matched in size only (issue 8) | Two controls: random directions (size-matched) and a uniform shrink (each token keeps its direction and gets the same length as the loud edit) |
| No correction for many comparisons (issue 11) | One main view per hypothesis; Holm correction across the four intervention layers |
| Named results are at a ceiling for accuracy | Named sets are compared on *change* across layers (the drop), not level |

**Not in study 2** (planned for later): the name-vs-familiarity shape sets and in-prompt name teaching (issues 1 and 2), a larger model, the Gemma check, and the transformed-copy task.

## 3. Design

### Items

Six sets, 300 items each, built by `python -m nwf.s2stimuli` with the same layout code as study 1. Items 0–99 of each set are **calibration**: the first 20 fit the loud directions, and all 100 may serve as extra training items for the trained probe, but no calibration item is ever scored in a result. Items 100–299 are the **test items**.

| Set | Shapes | Colours of the two pictures |
|---|---|---|
| `s2_named_fill` | the 8 named shapes, filled | different (as in study 1) |
| `s2_named_line` | the 8 named shapes, outlines | different |
| `s2_unnamed_fill` | 40 random blobs, filled | different |
| `s2_unnamed_line` | 40 squiggles with 30 bends | different |
| `s2_named_line_same` | the 8 named outlines | **the same** |
| `s2_unnamed_line_same` | 40 squiggles with 30 bends | **the same** |

The **main comparison** is `s2_named_line` vs `s2_unnamed_line`: both are line drawings, so fill plays no part. For the squiggles and blobs, item *i* draws its shapes from pool half *i* mod 2, so a probe trained on one half is tested on shapes it never saw.

### Models

Qwen3-VL-2B (base) and the paper's name-taught Qwen3-VL-2B, both in fp16 (fp16 passed study 1's precision check). Only the base model gets the interventions.

### What is measured on every test item

One forward pass per item records, at every layer input (29 stream indices):

1. **Same-shape probe (MaxSim)** in six views:
   - `full`: all tokens of each shape region (as in study 1);
   - `ink`: only tokens whose pixels inside the shape box are at least 10% ink;
   - `loud16_out` and `loud64_out`: after removing the top 16 or 64 principal directions of the image tokens at that layer (fitted on calibration items only);
   - `mean_out`: after subtracting each picture's own average token;
   - `ink_loud64_out`: ink tokens with the top 64 directions removed;
   - plus 10 seeded random 64-direction removals as the control.
2. **The average vector of each shape region**, for the trained probe.
3. **The state at the answer position** (the last token, where the letter is read).
4. **The four letter scores** (the model's answer).

The interventions run on the test items of the four core sets. At one layer at a time (3, 9, 15 and 21), the image tokens are changed in one of three ways, and the four letter scores are read:
- **loud64**: halve the top 64 directions;
- **random64**: halve 64 random directions;
- **uniform**: shrink each token to the length the loud edit gives, keeping its direction.

### Trained probe (`python -m nwf.analysis.layer_probe`)

- **Same-shape probe:** for each of the four candidates, the size of its difference from the REF region along each of 128 principal directions (fitted on the training items and scaled to equal spread), plus the average of those sizes. A learned weight for each direction gives every candidate a score; the four scores go through a softmax, and the weights are trained so the right candidate wins. The weight penalty is chosen on a split of the training items. Trained on one pool half and tested on the other, then the reverse.
- **Comparisons for the same-shape probe:** the same distance with every direction weighted equally (`untrained`), and the plain cosine between region averages (`cosine`). If the trained probe beats `untrained`, the model holds the shape in a few specific directions that a plain distance misses.
- **Answer-position probe:** a four-way classifier (same softmax set-up) predicting the right letter from the last token's state, with the same split, and a **shuffled-label control** trained on random answers.
- A shuffled-label control is not used for the same-shape probe: any weighting that prefers the closest candidate already solves the task, so a probe trained on random answers can still score high, and the control would mean nothing.

### Statistics

- Every number has a 95% bootstrap interval: 10,000 resamples of whole REF shapes, so repeated shapes don't count as independent.
- Layers are summarised as early, middle and late thirds of stream indices 3–28 (the first three are where Qwen adds extra vision features).
- **Drop** = early-third accuracy − middle-third accuracy.

## 4. Hypotheses and rules (fixed before data)

Main model: base Qwen. Main pair: `s2_named_line` vs `s2_unnamed_line`. "Interval above 0" means the 95% interval's lower end is above 0.

| | Hypothesis | Main measure | Supported if |
|---|---|---|---|
| **H1** | Unnamed information fades more than named | MaxSim `full` drop, unnamed − named | interval above 0 (below 0 = reversed) |
| **H2** | The fade is real loss, not a probe artefact | Trained same-shape probe drop on `s2_unnamed_line` | **lost** if drop ≥ 0.10 and interval above 0; **kept but reshaped** if drop < 0.05 and interval's upper end < 0.10; otherwise unclear |
| **H3** | Loud directions cover the unnamed information | Middle third: `loud64_out` − random removals, unnamed minus the same for named | the unnamed difference and the unnamed − named difference both have intervals above 0 (only the first = partly) |
| **H4** | Background colour drives the unnamed failure | Answer accuracy: `s2_unnamed_line_same` − `s2_unnamed_line` | ≥ 0.10 with interval above 0 (interval above 0 but smaller = small effect) |
| **H5** | The match reaches the answer position | Answer-position probe, late third, vs its control | reported for named and unnamed separately |
| **H6** | Turning loud directions down helps unnamed answers | Change in right-letter log-odds, loud64 − uniform and loud64 − random64, on `s2_unnamed_line` | both contrasts positive with Holm-adjusted p < 0.05 at the same layer (4 layers) |
| **H7** | Name teaching keeps unnamed information alive | Middle-third MaxSim `full`, name-taught − base, on `s2_unnamed_line` | interval above 0 |

Everything else (other views, sets, layers and both models' full curves) is shown as secondary and is not used for these verdicts. The rules are coded in `src/nwf/analysis/s2report.py` (`verdicts`) and `layer_probe.py`.

## 5. How to read the outcomes

- **H1 yes, H2 kept, H3 yes:** the information is still there but covered by loud directions. Then H6 decides whether a gentle edit fixes answers, which would be a training-free method.
- **H1 yes, H2 lost:** the model really discards unnamed information. Then H5 and H7 locate where and whether names prevent it.
- **H4 yes:** a large part of the "unnamed" failure is background colour. This must be reported, and it weakens the "no word" framing.
- **H1 no:** the fade seen in study 1 does not replicate on fresh items.

## 6. Running it

`kaggle/study2/run.py` runs everything in one Kaggle session (GPU T4 x2, internet on, secrets `GITHUB_TOKEN` and `HF_TOKEN`):

1. unit tests (`tests/test_s2.py`, `tests/test_loudness.py`);
2. building the items (about 10–15 min, CPU);
3. a **smoke run** on 6 items per set for both models, with checks: no NaN, correct shapes, named-shape probe ≥ 0.8, interventions actually change the scores. The run stops if any check fails;
4. the main run: base Qwen with interventions, then name-taught Qwen;
5. the analysis: both trained probes and the report, written to `/kaggle/working/analysis`.

**Time estimate (unverified):** about 4–5 GPU hours in total. The interventions are the largest part, about 9,600 extra forward passes. If the 11 h stop is reached, add that version's output as an input of the notebook (Add Input → Your Work) and run again: the script copies the items and finished chunks over and skips them.

## 7. Limitations known in advance

- One model family (Qwen3-VL-2B), one task (matching identical copies), fp16 on T4.
- MaxSim and the trained probe read region averages or token similarities; neither proves the model uses the information.
- The named sets reuse 8 shapes, so the trained probe on named sets is split by item, not by shape.
- The loud directions are fitted on 120 calibration items; a different fit could change `loud64_out` slightly.

## 8. Deviations log

| Date | Change | Reason |
|---|---|---|
| 2026-11-02 (before any study 2 data) | Same-shape probe changed from a ridge classifier on pairs with a shuffled-label control to a softmax over the four candidates, compared with the untrained distance and cosine | tests on fake data: the ridge version did worse than a plain distance, and its shuffled-label control landed near 0 or 1 by chance, so it could not serve as a control |
