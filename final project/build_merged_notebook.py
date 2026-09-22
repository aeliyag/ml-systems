#!/usr/bin/env python3
"""Build final-project-merged.ipynb from Clarisse base + Aeliya additions."""

import copy
import json
from pathlib import Path

ROOT = Path(__file__).parent
CLARISSE = ROOT / "final-project-clarisse.ipynb"
AELIYA = ROOT / "final-project-aeliya.ipynb"
OUT = ROOT / "final-project-merged.ipynb"


def lines(text: str) -> list[str]:
    return [line + "\n" for line in text.split("\n")]


BACKGROUND = """---

## Background & Key Terms (for readers new to networking)

This section defines the ideas used throughout the notebook. You do **not** need a networking background to follow the analysis. For implementation decisions, see [`design.md`](design.md). For a side-by-side diff of the two source notebooks, see [`COMPARISON.md`](COMPARISON.md).

### The real-world problem

When you stream video (Netflix, YouTube, etc.), the player chooses a **resolution** (e.g., 720p or 1080p) based on available bandwidth. Sometimes the network gets worse and the player **lowers** resolution so playback stays smooth. That drop is a **quality degradation** event.

We want to **predict these degradations a few seconds before they happen**, using measurements from network traffic, so a system could react early (reroute traffic, notify the user, request a lower bitrate proactively).

### Core vocabulary

| Term | Plain-language meaning |
|------|------------------------|
| **QoE (Quality of Experience)** | How good the video feels to the viewer (smooth playback, sharp picture, few stalls). |
| **Resolution** | Video sharpness, measured in pixels (240, 360, 480, 720, 1080). Higher = better quality, needs more bandwidth. |
| **Downswitch** | The stream **drops to a lower resolution** within the next Δ seconds. Primary horizon: **Δ = 30 seconds** (~3.9% positive rate). We also report Δ ∈ {10, 20}s as a sensitivity sweep. |
| **Session / video** | One continuous viewing period. Splits are grouped by **`video_id`** so the same video content never appears in both train and test. |
| **10-second window** | Each row is a snapshot of the network + player state during one 10s interval. |
| **Feature** | A numeric measurement the model uses as input (e.g., packet counts, retransmissions, chunk sizes). |
| **Label** | What we are trying to predict: will a downswitch happen soon? (`0` = no, `1` = yes). |

### Network "layers" (L3, L4, L7) — why tiers matter

Network data can be summarized at different depths. **Deeper summaries cost more CPU/time** to compute from raw packet captures (PCAP files). In this notebook, costs are **measured** from `netflix.pcap`, not assumed.

| Tier | Network layer (informal) | What it captures | Relative cost |
|------|--------------------------|------------------|---------------|
| **L3** | IP (network) | Basic packet/byte counts per flow | **Cheap** |
| **L4** | TCP (transport) | Retransmissions, RTT, window sizes — signs of congestion | **Measured expensive** |
| **L7** | Application (HTTP/video) | Chunk sizes, download timing — closer to what the video player sees | **Medium** (in our measurement) |
| **FULL** | L3 + L4 + L7 | All features together | **Highest** |

**Research idea:** Can we run a cheap model first and only compute expensive features when the cheap model is uncertain?

### Models we train (what they are and why)

| Model | What it is | Why we include it |
|-------|------------|-------------------|
| **Resolution-only** | Uses only the current resolution (e.g., "already at 240p") | Simple baseline: if you're at minimum quality, you can't downswitch further. |
| **Context-only** | Resolution + service | Second floor: how much signal exists without any network measurement? |
| **L3-only / L4-only / L7-only** | HistGB (or RF cross-check) on one feature tier | Shows how much each cost level buys in prediction quality. |
| **FULL** | All tier features | Upper bound for feature-rich models. |
| **Cascade** | L3 → L7 → L4 with isotonic calibration | Our **adaptive** system — pay full cost only when needed. |
| **Random escalation** | Randomly upgrades tiers at the same rate as the cascade | **Control experiment:** proves adaptive escalation beats blind random upgrades. |

### Evaluation metrics (what the numbers mean)

We treat this as **binary classification**: predict whether a downswitch will happen within Δ seconds.

| Metric | Meaning | Why we care |
|--------|---------|-------------|
| **Probability** | Model output from 0 to 1 ("how likely is a downswitch?") | Lets us trade off false alarms vs missed events by changing the threshold. |
| **Precision / Recall / F1** | Standard classification metrics at a chosen threshold | Show operational tradeoffs. |
| **PR curve / PR-AUC** | Precision–recall curve and its area | **Primary metric** for rare events (~3.9% at Δ=30). Random guessing ≈ base rate. |
| **Lift** | PR-AUC divided by base rate | How many times better than random. |
| **Lead time** | Seconds before a downswitch that the model first flags risk | Deployment metric: is the warning early enough to act? |
| **avg_cost / cost_ms** | Measured feature-extraction time per window | Connects ML quality to **systems cost**. |

**Intuition:** A model that always predicts "no downswitch" gets ~96% **accuracy** but is useless. **PR-AUC** rewards models that actually rank risky windows above safe ones. We also report metrics on the **`resolution > 240`** subset so structural zeros (already at minimum quality) do not inflate scores.
"""


def patch_setup(src: str) -> str:
    if "FIG_DIR" in src:
        return src
    insert = (
        "\nREPO_ROOT = Path.cwd()\n"
        "if not (REPO_ROOT / \"completed assignments\").exists():\n"
        "    REPO_ROOT = REPO_ROOT.parent\n"
        "FIG_DIR = REPO_ROOT / \"final project\" / \"figures\"\n"
        "FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
    )
    return src.replace("np.random.seed(RANDOM_STATE)\n", "np.random.seed(RANDOM_STATE)\n" + insert)


def add_savefig(src: str, filename: str, extra: str | None = None) -> str:
    needle = "plt.tight_layout(); plt.show()"
    saves = f'plt.savefig(FIG_DIR / "{filename}", dpi=120, bbox_inches="tight")'
    if extra:
        saves += f'\nplt.savefig(FIG_DIR / "{extra}", dpi=120, bbox_inches="tight")'
    replacement = f"plt.tight_layout()\n{saves}\nplt.show()"
    if filename in src:
        return src
    if needle in src:
        return src.replace(needle, replacement, 1)
    return src


def main() -> None:
    with CLARISSE.open() as f:
        nb = json.load(f)

    # Update title cell with merge provenance
    intro = nb["cells"][0]
    src = "".join(intro["source"])
    if "Source notebooks" not in src:
        addendum = (
            "\n\n**Source notebooks:** [`final-project-aeliya.ipynb`](final-project-aeliya.ipynb) · "
            "[`final-project-clarisse.ipynb`](final-project-clarisse.ipynb) · "
            "[`COMPARISON.md`](COMPARISON.md) · [`design.md`](design.md)\n\n"
            "> **New to networking or ML metrics?** Read **Background & Key Terms** (next section) before Part 1.\n"
        )
        intro["source"] = lines(src.rstrip() + addendum)

    # Insert Background section after intro
    background_cell = {"cell_type": "markdown", "metadata": {}, "source": lines(BACKGROUND.strip())}
    if not any("Background & Key Terms" in "".join(c.get("source", [])) for c in nb["cells"]):
        nb["cells"].insert(1, background_cell)

    # Patch setup cell (index 3 after background insert)
    for i, cell in enumerate(nb["cells"][:8]):
        if cell["cell_type"] == "code" and "find_data" in "".join(cell.get("source", [])):
            cell["source"] = lines(patch_setup("".join(cell["source"]).rstrip()))
            break

    # Figure exports on key plot cells
    figure_patches = [
        ("Precision-recall curves", "pr_curves.png", "confusion_matrices.png"),
        ("Quality vs. measured extraction cost", "quality_cost_curve.png", None),
        ("Reliability — full model", "calibration.png", None),
        ("Every model on one quality-cost plane", "summary_quality_cost.png", None),
    ]
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        for key, fname, extra in figure_patches:
            if key in src:
                cell["source"] = lines(add_savefig(src, fname, extra))
                src = "".join(cell["source"])

    # Update Part 14 sign-off
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell.get("source", []))
        if "Final sign-off" in src and "final-project-merged.ipynb" not in src:
            src = src.replace(
                "This notebook is Clarisse Cheung's independent implementation. Aeliya Grover implemented the same proposal separately; the two will be compared and merged.",
                "This notebook merges the independent implementations in `final-project-aeliya.ipynb` and `final-project-clarisse.ipynb`. Decision-by-decision provenance is in `COMPARISON.md`.",
            )
            src += (
                "\n- **Merged notebook.** Primary Δ=30s, video_id split, measured costs, and isotonic cascade calibration follow Clarisse's design. "
                "Background & Key Terms and figure exports follow Aeliya's design.\n"
                "- **Figures saved under** `final project/figures/` (pr_curves, confusion_matrices, quality_cost_curve, calibration, summary_quality_cost).\n"
            )
            cell["source"] = lines(src.rstrip())

    # Append figure listing cell if missing
    if not any("FIG_DIR.glob" in "".join(c.get("source", [])) for c in nb["cells"]):
        nb["cells"].append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": lines('print("Notebook complete. Figures:", sorted(FIG_DIR.glob("*.png")))'),
            }
        )

    # Clear outputs for cleaner commit (figures saved separately)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    with OUT.open("w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")

    print(f"Wrote {OUT} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
