# Phase 2 status — v3 is best at 0.6171 LB

## What v3 does (`submissions/phase2_v3_chead.csv`)

Two-stage classifier on segmented cells. Inputs come from Phase 1
segmentation (reused unchanged at 0.83 LB).

### Stage 1: subclass classifier (`runs/phase2_baseline`)

Per cell, build a feature vector:
- 1147 raw gene counts (spots inside the cell's mask, per gene)
- log1p, then per-cell L2 normalize
- append 2 stage coords: `global_x_um`, `global_y_um` (FOV offset + centroid)

Train a single LightGBM multiclass classifier (n_estimators=500, lr=0.05,
max_depth=6, num_leaves=31, early_stopping=30) on 41 subclasses
(40 named + `background`). Class imbalance handled with sklearn-style
balanced sample weights. ~5400 training cells from 60 train FOVs;
validated on 1030 cells from 10 held-out FOVs.

At inference: predict subclass per cell. Cells called `background` get
all-bg labels. Named cells get their `(supertype, class)` parents and
`cluster` child via deterministic majority rollout from the training
hierarchy table.

### Stage 2: per-subclass cluster heads (`runs/phase2_clusterheads`)

For each subclass with ≥20 training cells AND >1 cluster, train a
separate LightGBM head on the SAME 1149-dim features but predicting
`cluster_label` within that subclass. 17 heads total (out of 41
subclasses).

At inference: cells whose stage-1 subclass has a head get routed
through that head — its argmax cluster overrides the deterministic
rollout. `(supertype, class)` come from the cluster→higher lookup.

### What this buys you

- Stage 1 alone (v1 baseline): 0.61 LB
- Stage 1 + cluster heads (v3): **0.6171 LB**

Cluster heads add resolution at the cluster/supertype levels (48 unique
clusters predicted vs 25 from majority rollout). Class/subclass
predictions are unchanged between v1 and v3.

## What we've tried

| Sub | Knob | Val | LB | Verdict |
| --- | --- | --- | --- | --- |
| **v1** | subclass LGBM, majority rollout | 0.5554 | 0.61 | reference |
| v2 | + `--bg_threshold 0.5` (force low-conf to bg) | 0.5417 | 0.6012 | hurt as expected |
| **v3** | v1 + 17 cluster heads | 0.5560 | **0.6171** | **best** |
| v4 (E) | retrain with 7 rich features (count, n_genes, mask vol, image_row/col, global_x/y) | 0.5585 | 0.5642 | -0.046, sign flip |
| v5 (F-K5) | v1 features + K=5 same-FOV neighbor-mean log1p expression (1147 extra dim) | 0.5413 | 0.5768 | -0.040, regression |
| v6 (F-K3) | same as v5 but K=3 | 0.5667 | 0.6055 | -0.012, sign flip |
| v7 (G) | spot-dropout aug, 4 copies, keep_p~U(0.5,1.0) | 0.5669 | 0.5822 | -0.035, sign flip |
| v8 | binary bg-gate at T=0.5 AND v3 pipeline (gate is a necessary condition for named) | n/a | 0.56 | -0.06 vs v3 |

## Val→LB calibration is broken

Six submissions beyond v1 baseline, six different lessons:

| Knob | Val Δ | LB Δ | Direction |
| --- | --- | --- | --- |
| B bg_th | -0.014 | -0.009 | same |
| C heads | +0.001 | +0.007 | same (12× amplified) |
| E rich | +0.003 | **-0.046** | **flip** |
| F-K5 | -0.014 | -0.040 | same (3× amplified) |
| F-K3 | +0.011 | **-0.012** | **flip** |
| G drop | +0.012 | **-0.035** | **flip** |

**Every positive val Δ we've measured has flipped sign on LB.** Only
same-direction wins have been on knobs that were already worse on val.
The val set (held-out FOVs from the same training brain) is not
predictive of the test set (different brain region FOVs).

Practical implication: you can't trust val to rank knobs. Submitting
based on a +0.005–0.015 val gain has been net-negative every time.

### Phase 2.8: tested whether a "harder" val subset helps. It doesn't.

Computed per-FOV stats for all 70 train+val FOVs and 10 test FOVs:
- Test spots/cell median range: 30–178 (median 121). FOV_H is an
  extreme low at 29.5.
- Val spots/cell median range: 56–176 (median 110).
- Train: 41–191 (median 143). Val is already biased low vs train.

Built two "test-like" subsets of the existing val set:
- **hard-5**: FOV_140, 142, 127, 153, 158 (closest to test density)
- **hard-7**: + 110, 111

Re-evaluated all 7 submissions on each subset and computed rank
correlation against LB:

| Split | Spearman vs LB | p | Picks LB winner? |
| --- | --- | --- | --- |
| full10 | +0.00 | 1.00 | No (picks v7) |
| hard7 | +0.31 | 0.50 | No (picks v7) |
| hard5 | +0.39 | 0.38 | No (picks v7) |

All three splits pick v7_dropout as winner; LB winner is v3_chead
(v7 is 5th of 7 on LB). Hard subset improves correlation modestly but
not enough to be useful. The fundamental issue is train brain ≠ test
brain — no FOV subset of train can fix that.

**Conclusion: rely on LB as source of truth. Don't iterate on val gains.**

Stats files: `logs/fov_stats_train.csv`, `logs/fov_stats_test.csv`.
Subset files: `phase2_val_hard5.txt`, `phase2_val_hard7.txt`.

## Diagnostics done

### FOV_140 underperforms across every variant (~0.24 ARI vs median 0.65)

- 40 of 65 named cells in FOV_140 are `006 L4/5 IT CTX Glut`.
  Baseline recall on L4/5 IT: **3 of 56** caught. The model
  systematically misses this class.
- For FOV_140 L4/5 cells: P(bg)=0.90, P(L4/5)=0.02, margin 0.88.
  Model is *confidently* calling them bg, not uncertain.
- Marker check: panel HAS L4/5 markers (Rorb, Rspo1, Cux2, Slc17a7,
  Satb2, etc.), but FOV_140 L4/5 cells express them ~2–6× weaker than
  L4/5 cells in training:
  - Rorb (canonical L4 marker): 0.07 vs 0.40 (5.7× weaker)
  - Slc17a7 (pan-glutamatergic): 1.77 vs 3.95
  - Total spots/cell: 133 vs 187 (~30% sparser)
- Mask volumes are NORMAL (30k px). It's not over-segmentation —
  the cells are genuinely spot-sparse.
- L2 normalization can't recover marker signal that was below
  detection threshold to begin with.

So FOV_140 is a distribution-shift FOV at the spot-density level. The
spot-dropout aug experiment (v7) was designed to address this directly
but flipped sign on LB.

### Bg/named confusion (baseline)

- True non-bg → predicted bg: 28.7% (139 cells)
- True bg → predicted named: 41.4% (226 cells)
- Model isn't uniformly bg-biased; it's class-specific (L4/5 IT, L6 IT,
  L5 IT all have near-zero recall).

### LightGBM is severely overfit

- train_logloss 0.011 vs val_logloss 1.10 (100× gap)
- This is a structural property; regularization is on the followup list
  but hasn't been tried yet.

## What's NOT been tried

Listed in priority order. Cheap diagnostics first.

1. **LightGBM regularization sweep** on baseline: higher
   `min_data_in_leaf`, lower `num_leaves`, `lambda_l1/l2`. Lowest blast
   radius — same features, same pipeline. Realistic outcome: ±0.005 LB.
   Won't blow up like E/F/G. Could even stack with v3.
2. **Hierarchical bg-vs-not classifier** as stage 0 before subclass.
   Decouples the two error modes (false-bg, false-named). Hasn't been
   coded. Same data ceiling but might shift the operating point.
3. **kNN in expression space** as a second opinion. Less prone to
   confidence collapse than LightGBM. Ensemble candidate.
4. **Ensemble of baseline + cluster_heads + (something)** —
   `scripts/phase2/predict_ensemble.py` is wired up. Untested.
5. **Per-FOV stratified val** — pick a val split that mirrors test FOV
   diversity better, so val→LB calibration isn't broken in the first
   place. Diagnostic only; doesn't change the model.
6. **Spot-level smoothing** (kNN voting among same-cell spots).
   Post-hoc, no retraining.

## Files

- `runs/phase2_baseline/` — v1 model, prod
- `runs/phase2_clusterheads/` — v3's cluster heads, prod
- `runs/phase2_e/`, `runs/phase2_f/`, `runs/phase2_f_k3/`, `runs/phase2_g_dropout/` — diagnostic-only, kept
- `submissions/SUBMISSIONS.md` — full per-submission log with notes
- `handoff.md` — original phase 2 handoff (still mostly accurate)
- `scripts/phase2/predict_ensemble.py` — ensemble script (untested)
- `scripts/phase2/train_classifier.py` — supports `--extras`,
  `--neighbor_K`, `--dropout_copies`, `--dropout_p_min/max`
