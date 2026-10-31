"""Train linear probes on activations/ to predict the judge label (REFUSE vs COMPLY).

Default (development) mode fits one probe per position and layer on train and scores it on val.
--final refits the best val probe and scores it on test and heldout. Run it only once the
probe is settled, because test and heldout must not guide development.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
ACTIVATIONS = ROOT / "activations"
CS = [0.0001, 0.001, 0.01, 0.1, 1.0]
FOLDS = 5
KEY = ("scenario_id", "variant", "sample")
# The main probe reads the last prompt token, before any response text exists.
# response_last sees the written action, so it is only a sanity check.
MAIN_POSITION = "prompt_last"


def load(acts_name, labels_name):
    """Return activations [N, positions, layers, hidden], labels (1 = REFUSE), and meta rows."""
    data = torch.load(ACTIVATIONS / acts_name)
    labels = {
        tuple(row[k] for k in KEY): row["label"]
        for row in map(json.loads, (RESULTS / labels_name).read_text().splitlines())
    }
    keep = [i for i, m in enumerate(data["meta"]) if labels.get(tuple(m[k] for k in KEY)) in ("REFUSE", "COMPLY")]
    meta = [data["meta"][i] for i in keep]
    y = np.array([labels[tuple(m[k] for k in KEY)] == "REFUSE" for m in meta], dtype=int)
    return data["acts"][keep], y, meta, data["positions"]


def fit(x, y, groups):
    """Fit a standardized L2 logistic probe, choosing C by grouped cross-validation."""
    folds = list(GroupKFold(n_splits=FOLDS).split(x, y, groups))
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=CS, cv=folds, scoring="roc_auc", max_iter=2000),
    )
    return probe.fit(x, y)


def score(probe, x, y, meta):
    pred = probe.predict(x)
    prob = probe.predict_proba(x)[:, 1]
    result = {
        "n": len(y),
        "n_refuse": int(y.sum()),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "auroc": roc_auc_score(y, prob) if 0 < y.sum() < len(y) else None,
    }
    for level in (1, 2, 3):
        mask = np.array([m["difficulty"] == level for m in meta])
        result[f"accuracy_l{level}"] = accuracy_score(y[mask], pred[mask])
        result[f"n_refuse_l{level}"] = int(y[mask].sum())
    return result


def subset(acts, y, meta, splits):
    idx = [i for i, m in enumerate(meta) if m["split"] in splits]
    return acts[idx], y[idx], [meta[i] for i in idx]


def features(acts, position, layer):
    return acts[:, position, layer].float().numpy()


def develop(acts, y, meta, positions):
    """Fit a probe for every position and layer on train and score it on val."""
    train, val = subset(acts, y, meta, {"train"}), subset(acts, y, meta, {"val"})
    groups = [m["group_id"] for m in train[2]]
    rows = []
    for p, position in enumerate(positions):
        for layer in range(acts.shape[2]):
            probe = fit(features(train[0], p, layer), train[1], groups)
            row = {
                "position": position,
                "layer": layer,
                "C": float(probe[-1].C_[0]),
                **score(probe, features(val[0], p, layer), val[1], val[2]),
            }
            rows.append(row)
            print(f"{position:15s} layer {layer:2d}  val auroc {row['auroc']:.3f}  acc {row['accuracy']:.3f}", flush=True)
    return rows


def final(acts, y, meta, positions, position, layer):
    """Refit one probe on train and score it on test and heldout."""
    p = positions.index(position)
    train = subset(acts, y, meta, {"train"})
    probe = fit(features(train[0], p, layer), train[1], [m["group_id"] for m in train[2]])
    out = {"position": position, "layer": layer, "C": float(probe[-1].C_[0])}
    for split in ("val", "test", "heldout"):
        x, ys, ms = subset(acts, y, meta, {split})
        out[split] = score(probe, features(x, p, layer), ys, ms)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts", default="acts.pt", help="file name inside activations/")
    parser.add_argument("--labels", default="judge_labels.jsonl", help="file name inside results/")
    parser.add_argument("--output", default="probe_val.jsonl", help="file name inside results/")
    parser.add_argument("--final", action="store_true", help="score on test and heldout")
    parser.add_argument("--position", help="with --final: position to use. Defaults to prompt_last")
    parser.add_argument("--layer", type=int, help="with --final: layer to use. Defaults to the best val layer for the position")
    args = parser.parse_args()

    acts, y, meta, positions = load(args.acts, args.labels)
    if not args.final:
        rows = develop(acts, y, meta, positions)
        with (RESULTS / args.output).open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        best = max((r for r in rows if r["position"] == MAIN_POSITION), key=lambda r: r["auroc"])
        print(f"Best val probe: {best['position']} layer {best['layer']} (auroc {best['auroc']:.3f})")
        print(f"Wrote {len(rows)} probe results to {RESULTS / args.output}")
        return

    position = args.position or MAIN_POSITION
    layer = args.layer
    if layer is None:
        rows = [json.loads(line) for line in (RESULTS / args.output).read_text().splitlines()]
        layer = max((r for r in rows if r["position"] == position), key=lambda r: r["auroc"])["layer"]
    result = final(acts, y, meta, positions, position, layer)
    output = RESULTS / "probe_final.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
