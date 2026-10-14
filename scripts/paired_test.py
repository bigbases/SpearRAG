"""Paired per-question significance of ablations, per dataset and pooled over datasets ($0, no LLM).

Delta = Full - ablation on the same questions (positive = the removed component helps).
  per dataset : paired bootstrap over questions (10,000 resamples) + exact sign test
  pooled      : stratified bootstrap (resample within each dataset); macro = mean of per-dataset deltas
Per-question accuracy = (Contain-Acc + GPT-Acc) / 2, Medical GPT-Acc. Uses the first seed's answers.

    python scripts/paired_test.py --results results --ablations no_bridge no_subgraph_ppr
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spearrag.utils import normalize_answer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=str(ROOT / "results"))
ap.add_argument("--ablations", nargs="+", required=True)
ap.add_argument("--datasets", nargs="+", default=["2wiki", "hotpotqa", "musique", "medical"])
ap.add_argument("--n-boot", type=int, default=10000)
args = ap.parse_args()


def per_question(path, dataset):
    d = json.load(open(path))
    g = np.asarray(d["seed_avg_gpt_acc_per_query"], dtype=float)
    if dataset == "medical":
        return g, d["query_ids"]
    c = np.asarray([float(normalize_answer(a) in normalize_answer(p)) for p, a in zip(d["predictions"], d["gold_answers"])])
    return (c + g) / 2, d["query_ids"]


def sign_p(w, l):
    n, k = w + l, max(w, l)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


rng = np.random.default_rng(0)
for abl in args.ablations:
    print(f"\n## {abl}\n| scope | delta (pp) | 95% CI | wins / losses | sign-test p |\n|---|---|---|---|---|")
    deltas = {}
    for ds in args.datasets:
        full_f, abl_f = Path(args.results) / f"{ds}_spearrag.json", Path(args.results) / f"{ds}_spearrag_{abl}.json"
        if not (full_f.exists() and abl_f.exists()):
            print(f"| {ds} | - | missing result file | | |")
            continue
        (full, qf), (arm, qa) = per_question(full_f, ds), per_question(abl_f, ds)
        assert qf == qa, f"{ds}: question order differs"
        d = full - arm
        deltas[ds] = d
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.n_boot)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        w, l = int((d > 0).sum()), int((d < 0).sum())
        print(f"| {ds} (n={len(d)}) | {100*d.mean():+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {w} / {l} | {sign_p(w, l):.3f} |")
    if len(deltas) >= 2:
        boots = np.array([np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in deltas.values()])
                          for _ in range(args.n_boot)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        w = sum(int((d > 0).sum()) for d in deltas.values())
        l = sum(int((d < 0).sum()) for d in deltas.values())
        macro = np.mean([d.mean() for d in deltas.values()])
        print(f"| **pooled macro** ({len(deltas)} datasets) | **{100*macro:+.2f}** | [{100*lo:+.2f}, {100*hi:+.2f}] | {w} / {l} | {sign_p(w, l):.3f} |")
