# Design: Adaptive Feature Selection for Predicting Video Quality Degradation

**Implementation design for `final-project.ipynb`** · Clarisse Cheung · September 2026

This document records the implementation decisions behind the notebook. The
[proposal](project-proposal.md) states *what* we are investigating; this states *how*, and
why each choice was made over the alternatives. It exists so the two halves of this project
can be compared decision-by-decision rather than only by final numbers.

---

## 1. Environment and data

**Interpreter.** `/usr/local/bin/python3.11` — scikit-learn 1.5.2, pandas 2.2.3, numpy 2.0.2,
scapy 2.5.0, netml 0.7.1. No xgboost, lightgbm, dpkt, or pyarrow are installed, so the design
uses only what is present.

**Modeling data.** `completed assignments/data/video-qoe/video_dataset.pkl.xz` — 204,713
ten-second windows, 170 columns, 4,000 sessions, 64 homes, four services (Netflix, YouTube,
Twitch, Amazon), spanning December 2017 to February 2019. No null values anywhere. Loaded
directly from the compressed file (~9s); the uncompressed copy is gitignored, so we do not
materialize one.

**Cost data.** `completed assignments/data/video-qoe/netflix.pcap` — 141,471 packets over
8.3 minutes, 99.8% TCP port 443, ~87 bidirectional flows. Read end-to-end by
`RawPcapReader` in 0.07s, which means per-tier computation will dominate the measurement
rather than file I/O.

### Cleaning

| Action | Rows/cols affected | Reason |
|---|---|---|
| Drop `resolution ∉ {240,360,480,720,1080}` | 4,650 rows | Invalid resolutions, per Assignment 1 |
| Drop `index` | 1 column | Verified to be all zeros — carries no information |
| Treat `video_position` as a duplicate of `relative_timestamp` | 1 column | Verified bit-identical |

After cleaning: 200,063 windows across 3,996 sessions, with `relative_timestamp` perfectly
contiguous at 10s within every session.

---

## 2. Label construction

At time *t*, predict whether resolution drops during *(t, t+Δ]*:

```
y(t) = 1  iff  min( resolution(t+10), …, resolution(t+Δ) ) < resolution(t)
```

**Built by explicit time-keyed lookup, not by `shift()`.** A window is marked *unlabelable*
if any future window inside the horizon is missing, and unlabelable windows are excluded
rather than silently assumed negative. Contiguity happens to hold in this dataset, but the
label must not depend on that holding.

Labels are built for Δ ∈ {10, 20, 30} seconds. **Δ = 30 is primary** — it is the most
learnable (3.87% positives) and gives the longest useful warning, while still labeling 94%
of windows. Δ ∈ {10, 20} are reported as a sensitivity sweep.

Verified positive rates:

| Δ | Labelable windows | Positive rate | Sessions with ≥1 positive |
|---|---|---|---|
| 10s | 196,067 (98.0%) | 1.75% | 43.0% |
| 20s | 192,072 (96.0%) | 2.94% | 43.0% |
| 30s | 188,077 (94.0%) | 3.87% | 43.1% |

These reproduce the proposal's stated 1.7–3.9% and 43% exactly.

**Structural zeros.** Windows already at 240p cannot downswitch, and contain **zero**
positives at every Δ. They are kept in training and in pooled metrics because a deployed
system sees them, but every headline metric is *also* reported on the `resolution > 240`
subset. Without that second number, a model can look strong simply by detecting 240p.

---

## 3. Splits

**Primary split: grouped by `video_id`, 60/20/20, stratified on service.**

The proposal specifies a session-level split. We use a strictly stronger one, because the
data contains a leak a session split does not close: there are 3,996 sessions but only 2,096
distinct videos, and one video appears in 129 different sessions. Grouping by session alone
therefore places the same video content in both train and test. Grouping by video implies
grouping by session, so this satisfies the proposal's requirement and closes the leak at no
cost.

The **test set is untouched until Part 9.** The validation split does double duty: isotonic
calibration and cascade threshold selection.

**Secondary split: temporal**, on session start time, earliest 80% train / latest 20% test.

This split is **service-confounded and must be reported per service.** The late 20% of
sessions is 40% Twitch and 38% Amazon but only 2.8% YouTube, against 30% YouTube in the
early portion. Because Twitch carries 7.2% positives and YouTube 0.8%, a pooled temporal
number would measure the change in service mix and misreport it as drift. The pooled number
is reported with the confound named; the per-service numbers are what the drift claim rests on.

### Excluded from features

`session_id`, `video_id`, `home_id` (identifiers — memorizable), `index` (all zeros),
`absolute_timestamp` (retained as metadata for the temporal split only; as a feature it
invites memorizing capture time), `video_position` / `relative_timestamp` (see §4).

**Forward-leakage audit.** No L3/L4/L7 feature looks ahead. The `L7_allprev_*` and
`L7_all_prev_*` families are cumulative over *previous* windows only, which is past
information and therefore legitimate.

---

## 4. Feature tiers

Tiers are taken directly from the column-name prefixes, which the dataset already carries:

| Tier | Count | Content | Systems cost |
|---|---|---|---|
| L3 | 11 | throughput, byte/packet counts, parallel flows | IP headers only |
| L4 | 95 | RTT, bytes in flight, retransmits, receive window | per-flow TCP state |
| L7 | 55 | chunk sizes, chunk inter-arrival times | chunk inference in encrypted traffic |

11 + 95 + 55 = 161, plus 9 metadata columns = 170. Verified.

**Context features.** `resolution` and `service` (one-hot) are available to **every** tier at
zero marginal cost. Current resolution is itself the output of the Assignment 1 model, so
assuming it is available is realistic rather than generous; withholding it would cripple the
L3 tier and force every model to rediscover that 240p cannot fall. Because this risks
inflating results, a **no-context ablation** is run: all models refit on network features
alone, to separate network signal from resolution signal.

**`video_position` is excluded** from the feature set. It is causally legitimate — it is known
at time *t* — but it belongs to no protocol layer, so including it would blur the
cost hierarchy the entire project rests on. This is a deliberate cost to predictive
performance in exchange for a clean tier story.

---

## 5. Models and baselines

`RandomForestClassifier` and `HistGradientBoostingClassifier`. HistGB replaces the proposal's
generic "gradient boosting" because it is dramatically faster on 200k rows, handles NaN
natively, and calibrates well — and because xgboost is not installed in this environment.

| Model | Features | Role |
|---|---|---|
| Resolution-only | `resolution` | Floor. Every result reported as lift over this. |
| L3-only / L4-only / L7-only | one tier + context | Fixed-tier comparison (disjoint tiers) |
| RF-full, HistGB-full | L3+L4+L7 + context | Ceiling |
| Cascade | adaptive | The contribution |
| Random escalation | rate-matched | Control |

---

## 6. Cascade

**Stages are cumulative, not disjoint: L3 → L3+L4 → L3+L4+L7.** Once a feature has been paid
for it is not discarded, so the cascade's second stage is *not* the L4-only baseline. Both are
built; the difference between them is itself reportable.

**Escalation is parameterized by fraction, not by a raw probability band.** At 3.87%
positives, predicted probabilities cluster near zero, which makes a fixed `[lo, hi]` band both
unstable and awkward to rate-match. Instead, windows are ranked by uncertainty — distance from
the decision threshold in log-odds — and the top *k*% escalate. Sweeping *k* from 0 to 100
traces the quality–cost curve directly, and makes the random-escalation control **exactly**
rate-matched by construction rather than approximately. Since the proposal's central claim
depends on beating that control, the comparison should not rest on an approximate match.

**Calibration:** isotonic regression per stage, fit on the validation split
(`CalibratedClassifierCV(cv='prefit')`). Isotonic over Platt because there is ample validation
data (~40k windows) and no reason to assume a sigmoid link.

---

## 7. Cost measurement

Three extractors over `netflix.pcap`, built on `RawPcapReader` plus `struct` (no scapy packet
objects — their construction cost would swamp the tier differences we are trying to measure):

- **L3** — parse IP header; tally bytes, packets, throughput, parallel flows per 10s window.
- **L4** — additionally parse TCP headers and maintain per-flow seq/ack state: bytes in
  flight, RTT samples, retransmit detection, receive window.
- **L7** — additionally infer chunk boundaries by the burst-gap heuristic (an idle gap in the
  downstream direction above a threshold marks a boundary), then compute chunk size and
  inter-arrival statistics.

**Two numbers are reported from the same code**, because the project needs both and they
answer different questions:

1. **End-to-end per-tier time** — what it costs to deploy a monitor computing exactly this
   tier. Drives the fixed-tier comparison.
2. **Marginal escalation cost** — the added cost of stepping up one tier, with shared parsing
   factored out. Drives the cascade's cost accounting, since that is what escalation actually
   spends.

Plus the **time-to-availability model** (L7 features cannot exist until chunks have arrived),
plotted against effective warning time and labeled throughout as a model, not a measurement.

---

## 8. Evaluation

**Headline: PR-AUC**, because at <4% positives both accuracy and ROC-AUC look strong for a
model that catches almost nothing. Reported alongside: precision, recall, F1 and a confusion
matrix at a recall-matched threshold; per-service and pooled; and on the `resolution > 240`
subset as described in §2. **Lead time** — seconds before a downswitch that the session is
first flagged. Reliability diagrams, false-positive and false-negative analysis, and
session-split versus temporal-split comparison.

---

## 9. Reproducibility

Seeds fixed at module load. Library versions printed in Part 1. Target runtime for "restart
kernel and run all" is **10–15 minutes**: `RandomizedSearchCV` (~15 candidates, 3-fold,
scored on `average_precision`) over a stratified subsample, with final fits on the full
training set.

---

## 10. Known risks

- **YouTube has only 0.8% positives at Δ=30.** Per-service PR-AUC for YouTube may be too
  noisy to interpret; it will be reported with its positive count so that noise is visible.
- **The burst-gap chunk heuristic is an approximation** of however the dataset's L7 features
  were originally produced. It is used to measure *cost*, not to reproduce the feature values,
  so exact agreement is not required — but the distinction is stated wherever L7 cost appears.
- **A fixed tier may dominate the cascade.** The proposal commits to reporting that outcome if
  it occurs, and this design does not hedge against it.

---

## 11. Merged decisions (September 2026)

The merged submission notebook is [`final-project-merged.ipynb`](final-project-merged.ipynb).
Side-by-side provenance is in [`COMPARISON.md`](COMPARISON.md). Source snapshots:

- [`final-project-clarisse.ipynb`](final-project-clarisse.ipynb) — commit `066608d`
- [`final-project-aeliya.ipynb`](final-project-aeliya.ipynb) — Aeliya's independent implementation

| Decision | Merged choice | Primary source |
|----------|---------------|----------------|
| Primary Δ | 30s (+ 10s/20s sensitivity) | Clarisse |
| Train/val/test split | Grouped by `video_id`, 60/20/20 | Clarisse |
| Label construction | Explicit time-keyed lookup | Clarisse |
| Cascade order | L3 → L7 → L4 (cost-measured) | Clarisse |
| Escalation policy | Top-k% uncertainty ranking | Clarisse |
| Calibration | Isotonic per cascade stage | Clarisse |
| Hyperparameter tuning | `RandomizedSearchCV` | Clarisse |
| Cost model | Measured from `netflix.pcap` | Clarisse |
| Reader documentation | Background & Key Terms section | Aeliya |
| Figure exports | Saved under `figures/` | Aeliya |
| Reproducibility tooling | `run_notebook.py`, `build_merged_notebook.py` | Aeliya |

**Validation:** Re-run `final-project-merged.ipynb` (or `run_notebook.py`) in the course
environment before submission. Figures under `figures/` must be regenerated from the merged
notebook; do not reuse numbers from either snapshot without re-running on the merged settings.
