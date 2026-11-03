"""Bootstrap 95% confidence intervals, resampling scenario groups (group_id) with replacement.

Twins and related scenarios share a group_id and are not independent, so whole groups are
resampled. Intervals are percentile intervals over REPS resamples, seed 0.

1. Behavior: REFUSE rate by level and variant, and the harmful minus benign gap, on
   development data (train + val) and on all 600 responses. Uses results/judge_labels.jsonl.
2. Prediction: AUROC and balanced accuracy for the probe and every baseline per split and
   subset, plus the paired difference probe minus each baseline. Uses
   results/predictions_final.jsonl from src/predictions.py.

Writes results/bootstrap.json.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
REPS = 2000
SEED = 0
MODELS = ("probe", "majority", "harmful_rule", "tfidf_prompt", "tfidf_response")
SUBSETS = {
    "all": lambda r: True,
    "l2": lambda r: r["difficulty"] == 2,
    "l3": lambda r: r["difficulty"] == 3,
    "l23": lambda r: r["difficulty"] >= 2,
    "l23_harmful": lambda r: r["difficulty"] >= 2 and r["variant"] == "harmful",
}


def resamples(rows, rng):
    """Yield REPS lists of row indices, each made by resampling group_ids with replacement."""
    by_group = defaultdict(list)
    for i, r in enumerate(rows):
        by_group[r["group_id"]].append(i)
    groups = list(by_group)
    for _ in range(REPS):
        picked = rng.choice(len(groups), size=len(groups), replace=True)
        yield [i for g in picked for i in by_group[groups[g]]]


def interval(estimate, samples):
    samples = [s for s in samples if s is not None and not np.isnan(s)]
    lo, hi = np.percentile(samples, [2.5, 97.5]) if samples else (None, None)
    return {"estimate": estimate, "lo": None if lo is None else float(lo), "hi": None if hi is None else float(hi),
            "valid_reps": len(samples)}


def auroc(y, prob):
    return roc_auc_score(y, prob) if 0 < y.sum() < len(y) else None


def balanced(y, pred):
    return balanced_accuracy_score(y, pred) if 0 < y.sum() < len(y) else None


def behavior(rows, rng):
    """REFUSE rate per level and variant, and the harmful minus benign gap per level."""
    refuse = np.array([r["label"] == "REFUSE" for r in rows], dtype=float)
    level = np.array([r["difficulty"] for r in rows])
    harmful = np.array([r["variant"] == "harmful" for r in rows])

    def stats(idx):
        idx = np.asarray(idx)
        out = {}
        for lv in (1, 2, 3):
            m = level[idx] == lv
            h = refuse[idx][m & harmful[idx]].mean() if (m & harmful[idx]).any() else np.nan
            b = refuse[idx][m & ~harmful[idx]].mean() if (m & ~harmful[idx]).any() else np.nan
            out[f"l{lv}_harmful"], out[f"l{lv}_benign"], out[f"l{lv}_gap"] = h, b, h - b
        return out

    point = stats(range(len(rows)))
    boots = [stats(idx) for idx in resamples(rows, rng)]
    n = {f"l{lv}": int((level == lv).sum() // 2) for lv in (1, 2, 3)}
    return {"pairs_per_level": n,
            **{k: interval(float(v), [b[k] for b in boots]) for k, v in point.items()}}


def prediction(rows, rng):
    """AUROC and balanced accuracy with CIs per model, and paired differences vs the probe."""
    out = {}
    for split in ("val", "test", "heldout"):
        for subset, keep in SUBSETS.items():
            sel = [r for r in rows if r["split"] == split and keep(r)]
            y = np.array([r["label"] == "REFUSE" for r in sel], dtype=int)
            probs = {m: np.array([r[f"{m}_prob"] for r in sel]) for m in MODELS}
            preds = {m: np.array([r[f"{m}_pred"] for r in sel]) for m in MODELS}

            def metrics(idx):
                idx = np.asarray(idx)
                res = {}
                for m in MODELS:
                    res[f"{m}_auroc"] = auroc(y[idx], probs[m][idx])
                    res[f"{m}_balanced_accuracy"] = balanced(y[idx], preds[m][idx])
                for m in MODELS[1:]:
                    for metric in ("auroc", "balanced_accuracy"):
                        a, b = res[f"probe_{metric}"], res[f"{m}_{metric}"]
                        res[f"probe_minus_{m}_{metric}"] = None if a is None or b is None else a - b
                return res

            point = metrics(range(len(sel)))
            boots = [metrics(idx) for idx in resamples(sel, rng)]
            out.setdefault(split, {})[subset] = {
                "n": len(sel), "n_refuse": int(y.sum()),
                **{k: interval(v, [b[k] for b in boots]) for k, v in point.items()},
            }
    return out


def fmt(ci):
    if ci["estimate"] is None:
        return "n/a"
    if ci["lo"] is None:
        return f"{ci['estimate']:.2f}"
    return f"{ci['estimate']:.2f} [{ci['lo']:.2f}, {ci['hi']:.2f}]"


def main():
    rng = np.random.default_rng(SEED)
    labels = [json.loads(line) for line in (RESULTS / "judge_labels.jsonl").read_text().splitlines()]
    preds = [json.loads(line) for line in (RESULTS / "predictions_final.jsonl").read_text().splitlines()]

    result = {
        "reps": REPS, "seed": SEED, "resample_unit": "group_id",
        "behavior_dev": behavior([r for r in labels if r["split"] in ("train", "val")], rng),
        "behavior_all": behavior(labels, rng),
        "prediction": prediction(preds, rng),
    }
    (RESULTS / "bootstrap.json").write_text(json.dumps(result, indent=2) + "\n")

    for name in ("behavior_dev", "behavior_all"):
        b = result[name]
        print(f"\n{name}: REFUSE rate [95% CI]")
        for lv in (1, 2, 3):
            print(f"  level {lv}: harmful {fmt(b[f'l{lv}_harmful'])}  benign {fmt(b[f'l{lv}_benign'])}  gap {fmt(b[f'l{lv}_gap'])}")
    for split in ("test", "heldout"):
        print(f"\n{split}: AUROC [95% CI]")
        for subset in ("all", "l23", "l23_harmful"):
            p = result["prediction"][split][subset]
            print(f"  {subset:12s} " + "  ".join(f"{m} {fmt(p[f'{m}_auroc'])}" for m in ("probe", "harmful_rule", "tfidf_prompt")))
            print(f"  {'':12s} probe - tfidf_prompt {fmt(p['probe_minus_tfidf_prompt_auroc'])}  "
                  f"probe - harmful_rule {fmt(p['probe_minus_harmful_rule_auroc'])}")
    print("\nWrote results/bootstrap.json")


if __name__ == "__main__":
    main()
