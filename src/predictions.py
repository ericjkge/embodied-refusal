"""Save per-example predictions of the frozen probe and baselines, plus confusion matrices.

Refits exactly the configurations scored by `probe.py --final --layer 29` and
`baselines.py --final` (same code, same data, deterministic solvers), so nothing is tuned.
Checks that the refit reproduces results/probe_final.json and results/baselines_final.json.

Writes:
  results/predictions_final.jsonl  one row per val/test/heldout response, with every model's
                                   probability and prediction (threshold 0.5)
  results/confusion_final.json     confusion matrix and per-class precision/recall per model,
                                   split, and subset
"""
import json

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import baselines
import probe

LAYER = 29
POSITION = probe.MAIN_POSITION
SPLITS = ("val", "test", "heldout")
MODELS = ("probe",) + baselines.BASELINES
SUBSETS = {
    "all": lambda r: True,
    "l1": lambda r: r["difficulty"] == 1,
    "l2": lambda r: r["difficulty"] == 2,
    "l3": lambda r: r["difficulty"] == 3,
    "l23": lambda r: r["difficulty"] >= 2,
    "l23_harmful": lambda r: r["difficulty"] >= 2 and r["variant"] == "harmful",
}
KEY = probe.KEY


def probe_predictions():
    """Refit the frozen probe on train and return {key: (prob, pred)} for every other split."""
    acts, y, meta, positions = probe.load("acts.pt", "judge_labels.jsonl")
    p = positions.index(POSITION)
    x, ys, ms = probe.subset(acts, y, meta, {"train"})
    model = probe.fit(probe.features(x, p, LAYER), ys, [m["group_id"] for m in ms])
    out = {}
    for split in SPLITS:
        x, ys, ms = probe.subset(acts, y, meta, {split})
        feats = probe.features(x, p, LAYER)
        for m, prob, pred in zip(ms, model.predict_proba(feats)[:, 1], model.predict(feats)):
            out[tuple(m[k] for k in KEY)] = (float(prob), int(pred))
    return out


def baseline_predictions(rows, y):
    """Refit every baseline on train and return {name: {key: (prob, pred)}}."""
    train_rows, train_y = baselines.subset(rows, y, {"train"})
    groups = [r["group_id"] for r in train_rows]
    out = {}
    for name in baselines.BASELINES:
        model = baselines.make(name).fit(train_rows, train_y, groups)
        eval_rows = [r for r in rows if r["split"] in SPLITS]
        pred, prob = model.predict(eval_rows)
        out[name] = {tuple(r[k] for k in KEY): (float(pr), int(pd)) for r, pr, pd in zip(eval_rows, prob, pred)}
    return out


def confusion(y, pred):
    """Confusion matrix (rows = judge label, columns = prediction) and per-class precision/recall."""
    cm = confusion_matrix(y, pred, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        "matrix": {"COMPLY": {"COMPLY": int(cm[0, 0]), "REFUSE": int(cm[0, 1])},
                   "REFUSE": {"COMPLY": int(cm[1, 0]), "REFUSE": int(cm[1, 1])}},
        "COMPLY": {"precision": precision[0], "recall": recall[0], "f1": f1[0], "support": int(support[0])},
        "REFUSE": {"precision": precision[1], "recall": recall[1], "f1": f1[1], "support": int(support[1])},
    }


def check(rows, final_probe, final_baselines):
    """Make sure the refit reproduces the scored final runs."""
    for split in SPLITS:
        sel = [r for r in rows if r["split"] == split]
        y = np.array([r["label"] == "REFUSE" for r in sel], dtype=int)
        for name, expected in [("probe", final_probe[split])] + [(b["baseline"], b[split]) for b in final_baselines]:
            prob = np.array([r[f"{name}_prob"] for r in sel])
            pred = np.array([r[f"{name}_pred"] for r in sel])
            got = baselines.score(pred, prob, y, sel)
            for key in ("auroc", "balanced_accuracy", "auroc_l23"):
                if expected[key] is not None and abs(got[key] - expected[key]) > 1e-9:
                    raise SystemExit(f"{name} {split} {key}: refit {got[key]} != final {expected[key]}")


def main():
    rows, y = baselines.load("judge_labels.jsonl")
    probe_out = probe_predictions()
    base_out = baseline_predictions(rows, y)

    out_rows = []
    for r in rows:
        if r["split"] not in SPLITS:
            continue
        key = tuple(r[k] for k in KEY)
        row = {k: r[k] for k in ("scenario_id", "variant", "sample", "hazard_type", "difficulty", "group_id", "split", "label")}
        row["probe_prob"], row["probe_pred"] = probe_out[key]
        for name in baselines.BASELINES:
            row[f"{name}_prob"], row[f"{name}_pred"] = base_out[name][key]
        out_rows.append(row)

    check(out_rows,
          json.loads((probe.RESULTS / "probe_final.json").read_text()),
          json.loads((probe.RESULTS / "baselines_final.json").read_text()))

    with (probe.RESULTS / "predictions_final.jsonl").open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    report = {}
    for model in MODELS:
        for split in SPLITS:
            for subset, keep in SUBSETS.items():
                sel = [r for r in out_rows if r["split"] == split and keep(r)]
                y_true = np.array([r["label"] == "REFUSE" for r in sel], dtype=int)
                pred = np.array([r[f"{model}_pred"] for r in sel])
                report.setdefault(model, {}).setdefault(split, {})[subset] = confusion(y_true, pred)
    (probe.RESULTS / "confusion_final.json").write_text(json.dumps(report, indent=2) + "\n")

    for split in ("test", "heldout"):
        print(f"\n{split}, all levels: confusion (judge rows x prediction cols) and REFUSE/COMPLY precision, recall")
        for model in MODELS:
            c = report[model][split]["all"]
            m = c["matrix"]
            print(f"{model:15s} C->C {m['COMPLY']['COMPLY']:3d} C->R {m['COMPLY']['REFUSE']:3d} "
                  f"R->C {m['REFUSE']['COMPLY']:3d} R->R {m['REFUSE']['REFUSE']:3d}   "
                  f"REFUSE P {c['REFUSE']['precision']:.2f} R {c['REFUSE']['recall']:.2f}   "
                  f"COMPLY P {c['COMPLY']['precision']:.2f} R {c['COMPLY']['recall']:.2f}")
    print(f"\nRefit matches the final runs. Wrote {len(out_rows)} rows to results/predictions_final.jsonl "
          "and results/confusion_final.json")


if __name__ == "__main__":
    main()
