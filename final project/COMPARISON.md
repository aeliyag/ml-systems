# Final Project Notebook Comparison

**Authors:** Aeliya Grover · Clarisse Cheung  
**Snapshots compared:**
- [`final-project-aeliya.ipynb`](final-project-aeliya.ipynb) — Aeliya's local implementation (Δ=10 primary)
- [`final-project-clarisse.ipynb`](final-project-clarisse.ipynb) — Clarisse's pushed implementation (`066608d`, Δ=30 primary)
- **Merged target:** [`final-project-merged.ipynb`](final-project-merged.ipynb)

Both notebooks diverged from the same blank template (`3fb27b6`). They are parallel implementations, not sequential edits.

---

## Executive summary

| Dimension | Aeliya | Clarisse | Merged choice |
|-----------|--------|----------|---------------|
| Primary Δ | 10s (1.7% positives) | **30s (3.9% positives)** | **30s** (+ 10s/20s sweep) |
| Split | `session_id` grouped | **`video_id` grouped** | **video_id** |
| Reader docs | Background & Key Terms | `design.md` | **Both** |
| Tuning | GridSearchCV (`n_estimators`) | RandomizedSearchCV | **RandomizedSearchCV** |
| Calibration | Basic curve only | Isotonic per cascade stage | **Isotonic** |
| Cost model | Assumed tier weights (1/4/10) | Measured from PCAP timing | **Measured** |
| Figures | Exported to `figures/` | Inline only | **Exported** |
| Part 14 | Short checklist | Detailed checklist + sign-off | **Combined** |

---

## Part-by-part comparison

### Part 1: Setup & Configuration

| | Aeliya | Clarisse | Verdict |
|---|--------|----------|---------|
| Status | **Different** | **Different** | Merge both strengths |
| Paths | Fixed `REPO_ROOT` + `video_dataset.pkl` | `find_data()` auto-discovery + `.pkl.xz` | Use Clarisse's `find_data()` |
| Seeds | `RANDOM_STATE = 42` | `RANDOM_STATE = 42` | Same |
| Figures dir | `FIG_DIR` created | Missing | Add `FIG_DIR` from Aeliya |
| Styling | seaborn whitegrid | Custom matplotlib theme | Keep Clarisse's theme |

**Merge:** Clarisse base + `FIG_DIR` export path.

---

### Part 2: Data Loading & Integrity Checks

| | Aeliya | Clarisse | Verdict |
|---|--------|----------|---------|
| Status | **Same goal, different depth** | **Same goal, different depth** | Merge |
| Cleaning | Drop invalid resolution, keep metadata | Same + drop `index`, dedupe `video_position` | **Clarisse** (more thorough) |
| Integrity | Basic shape/null checks | Session contiguity, service balance plots | **Clarisse** |
| Load path | Uncompressed `.pkl` | Compressed `.pkl.xz` | **Clarisse** |

**Merge:** Clarisse's loading and integrity checks.

---

### Part 3: Labels, Splits & Leakage Checks

| | Aeliya | Clarisse | Verdict |
|---|--------|----------|---------|
| Status | **Different** | **Different** | Critical fork |
| Primary Δ | **10s** | **30s** | **30s** (merged) |
| Label build | `groupby` + `shift()`-style | Explicit time-keyed lookup | **Clarisse** (safer) |
| Split | `session_id` GroupShuffleSplit | **`video_id` grouped 60/20/20** | **Clarisse** (closes video leak) |
| Leakage test | Temporal split | Shuffled-label PR-AUC + temporal + per-service | **Clarisse** |
| Structural zeros | Mentioned in baseline | **`resolution > 240` subset metrics** | **Clarisse** |
| Reader prose | Strong PR-AUC vs accuracy explanation | Strong inline metric discussion | **Both** |

**Merge:** Clarisse methodology; keep Aeliya's accessible metric explanations in Background section.

---

### Part 4: Feature Tiers (L3 / L4 / L7)

| | Aeliya | Clarisse | Verdict |
|---|--------|----------|---------|
| Status | **Different cost assumptions** | **Different cost assumptions** | Clarisse |
| Tier defs | L3/L4/L7 column prefixes | Same + measured extraction timing | **Clarisse** |
| Cost | Assumed weights {1, 4, 10} | **Timed over `netflix.pcap`** — L4 > L7 | **Clarisse** |

**Merge:** Clarisse's measured cost hierarchy (L4 is expensive tier, cascade order L3→L7→L4).

---

### Part 5: Baselines & Control Experiment Design

| | Aeliya | Clarisse | Verdict |
|---|--------|----------|---------|
| Status | **Similar** | **Similar** | Merge |
| Baselines | Resolution-only | Resolution-only + **context-only** (service+resolution) | **Clarisse** |
| Control | Random escalation at matched rate | Random escalation, **fraction-parameterized** | **Clarisse** |

**Merge:** Clarisse's richer baseline set.

---

### Part 6: Train Multiple Models

| | Aeliya | Clarisse | Verdict |
|---|--------|----------|---------|
| Status | **Different results** | **Different results** | Clarisse structure |
| Models | RF-FULL, HistGB-FULL, L3/L4/L7 RF | HistGB workhorse + RF cross-check | **Clarisse** |
| Best PR-AUC | RF-FULL **0.440** (Δ=10, session split) | HistGB-FULL **0.715** (Δ=30, video split) | Not comparable until re-run |

**Merge:** Clarisse model training pipeline.

---

### Part 7: Adaptive Cascade

| | Aeliya | Clarisse | Verdict |
|---|--------|----------|---------|
| Status | **Different** | **Different** | Clarisse |
| Stages | L3 → L3+L4 → FULL | **L3 → L7 → L4** (cost-driven order) | **Clarisse** |
| Threshold | Fixed low/high probability band | **Top-k% uncertainty ranking** | **Clarisse** |
| Calibration | None per stage | **Isotonic per stage** | **Clarisse** |
| Cascade PR-AUC | 0.336 (Δ=10) | ~0.690 (97% of full at Δ=30) | Re-run in merged |

**Merge:** Clarisse cascade design entirely.

---

### Part 8: Hyperparameter Tuning & Validation

| | Aeliya | Clarisse | Verdict |
|---|--------|----------|---------|
| Status | **Different** | **Different** | Clarisse |
| Method | GridSearchCV on `n_estimators` | RandomizedSearchCV, broader | **Clarisse** |
| Validation | Session-grouped val fold | Val fold for calibration + thresholds | **Clarisse** |

**Merge:** Clarisse tuning.

---

### Part 9: Prediction Evaluation

| | Aeliya | Clarisse | Verdict |
|---|--------|----------|---------|
| Status | **Different depth** | **Different depth** | Merge |
| Metrics | PR-AUC, F1, confusion, per-service, lead time | Same + lift over base rate + ROC comparison plot | **Clarisse** |
| Subset | No `resolution > 240` split | **Reports pooled + resolution>240** | **Clarisse** |
| Figures | Saved PNGs | Inline only | **Aeliya export** on Clarisse plots |

**Merge:** Clarisse evaluation + Aeliya figure exports.

---

### Part 10: Systems-Cost Evaluation

| | Aeliya | Clarisse | Verdict |
|---|--------|----------|---------|
| Status | **Different** | **Different** | Clarisse |
| Cost | Relative tier weights | **Measured ms/window from PCAP** | **Clarisse** |
| Curve | Quality–cost sweep (assumed costs) | Pareto frontier with measured costs | **Clarisse** |
| Extra | — | Part 10.5 time-to-availability model (labelled as model) | Keep with label |

**Merge:** Clarisse systems evaluation.

---

### Part 11: Error Analysis & Robustness Checks

| | Aeliya | Clarisse | Verdict |
|---|--------|----------|---------|
| Status | **Different** | **Different** | Clarisse |
| Drift | Temporal session split | Temporal + **per-service confound analysis** | **Clarisse** |
| Calibration | Single RF-FULL curve | Full + cascade stage reliability diagrams | **Clarisse** |
| FP/FN | Basic analysis | Detailed false positive/negative breakdown | **Clarisse** |

**Merge:** Clarisse error analysis.

---

### Part 12: Cross-Model Comparison & Interpretation

| | Aeliya | Clarisse | Verdict |
|---|--------|----------|---------|
| Status | **Similar** | **Similar** | Clarisse |
| Summary | Markdown table | `SUMMARY` dataframe + quality–cost plane | **Clarisse** |
| Key insight | Cascade doesn't match FULL at low escalation | **95% quality at 51% cost; L3+L7 nearly as simple** | **Clarisse** (more nuanced) |

**Merge:** Clarisse synthesis.

---

### Part 13: Conclusions

| | Aeliya | Clarisse | Verdict |
|---|--------|----------|---------|
| Status | **Different** | **Different** | Re-write after merged run |
| Answer | Partial cost savings, cascade below FULL | **Substantially yes, bounded win** | **Clarisse framing** |
| Honesty | Notes cascade gap | Notes cost hierarchy reversal, L3+L7 simplicity | **Clarisse** |

**Merge:** Use Clarisse conclusions after merged notebook re-run validates numbers.

---

### Part 14: Reproducibility & Submission Checklist

| | Aeliya | Clarisse | Verdict |
|---|--------|----------|---------|
| Status | **Different depth** | **Different depth** | **Combined** |
| Aeliya | Short markdown checklist + figure list | — | Keep items |
| Clarisse | — | Detailed programmatic checks + grader sign-off | Keep items |

**Merge:** Clarisse programmatic checklist + Aeliya figure-export verification + merge provenance note.

---

## Numeric results (not directly comparable)

These numbers use **different Δ and splits**. Do not compare across columns until both run on merged settings (Δ=30, video_id split).

| Model | Aeliya PR-AUC (Δ=10, session) | Clarisse PR-AUC (Δ=30, video) |
|-------|-------------------------------|-------------------------------|
| Resolution-only | ~0.027 | 0.066 |
| L3 | — | 0.558 |
| L4 | ~0.344 | 0.640 |
| L7 | — | 0.677 |
| RF/HistGB FULL | **0.440** | **0.715** |
| Cascade | 0.336 | ~0.690 (97% of full) |
| Random escalation | similar to cascade | beaten by cascade (+0.054 to +0.119) |

**Aeliya avg_cost:** relative weights (L3=1, FULL=10).  
**Clarisse cost:** measured ms/window from PCAP extractors.

---

## Merge provenance

| Element | Source |
|---------|--------|
| Notebook structure, Δ=30, splits, models, cascade, evaluation | Clarisse |
| Background & Key Terms (updated for Δ=30) | Aeliya |
| Figure exports to `figures/` | Aeliya |
| `design.md` engineering doc | Clarisse |
| Part 14 combined checklist | Both |
| `run_notebook.py` validation target | Merged notebook |

See also: [`design.md`](design.md) §Merged decisions appendix.
