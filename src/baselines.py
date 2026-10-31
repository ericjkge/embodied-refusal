"""Text and trivial baselines for predicting the judge label (REFUSE vs COMPLY).

Each baseline is trained on train, with C chosen by grouped cross-validation on group_id
(as in probe.py), and scored on val. --final refits the same baselines on train and scores
them on test and heldout. Run it only once, alongside probe.py --final, because test and
heldout must not guide development.

Baselines and the information each one receives:
  majority          nothing: always predicts the most common train label
  harmful_variant   the harmful/benign tag: predicts REFUSE for every harmful prompt
  tfidf_prompt      scene + instruction text (the same input the prompt_last probe reads)
  tfidf_reasoning   the response's Reasoning text, with the Action line removed
  tfidf_response    the full response, including the action (sanity check, like response_last)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
CS = [0.01, 0.1, 1.0, 10.0, 100.0]
FOLDS = 5
BOOTSTRAPS = 1000
SEED = 0
# Subsets reported for every baseline. Levels 2 and 3 are the main result, because level 1
# is nearly all refusals. "harmful" asks whether a method predicts which harmful prompts get refused.
SUBSETS = {
    "all": lambda r: True,
    "l1": lambda r: r["difficulty"] == 1,
    "l2": lambda r: r["difficulty"] == 2,
    "l3": lambda r: r["difficulty"] == 3,
    "l23": lambda r: r["difficulty"] in (2, 3),
    "harmful": lambda r: r["variant"] == "harmful",
}


def prompt_text(row):
    return row["scene"] + "\n" + row["instruction"]


def reasoning_text(row):
    return row["response"].split("Action:")[0]


def response_text(row):
    return row["response"]


TEXT_BASELINES = {
    "tfidf_prompt": prompt_text,
    "tfidf_reasoning": reasoning_text,
    "tfidf_response": response_text,
}


def load(labels_name):
    """Return judge label rows with a REFUSE or COMPLY label, and labels (1 = REFUSE)."""
    rows = [json.loads(line) for line in (RESULTS / labels_name).read_text().splitlines()]
    rows = [r for r in rows if r["label"] in ("REFUSE", "COMPLY")]
    y = np.array([r["label"] == "REFUSE" for r in rows], dtype=int)
    return rows, y


def subset(rows, y, splits):
    idx = [i for i, r in enumerate(rows) if r["split"] in splits]
    return [rows[i] for i in idx], y[idx]


def fit_text(texts, y, groups):
    """Fit TF-IDF + L2 logistic regression, choosing C by grouped cross-validation.

    The vectorizer is inside the pipeline, so each fold fits its own vocabulary and IDF.
    """
    folds = list(GroupKFold(n_splits=FOLDS).split(texts, y, groups))
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=2000),
    )
    search = GridSearchCV(pipeline, {"logisticregression__C": CS}, cv=folds, scoring="roc_auc")
    return search.fit(texts, y)


class Majority:
    def fit(self, rows, y):
        self.label = int(y.mean() > 0.5)
        return self

    def predict_proba(self, rows):
        return np.full(len(rows), float(self.label))


class HarmfulVariant:
    def fit(self, rows, y):
        return self

    def predict_proba(self, rows):
        return np.array([r["variant"] == "harmful" for r in rows], dtype=float)


class Text:
    def __init__(self, to_text):
        self.to_text = to_text

    def fit(self, rows, y):
        self.search = fit_text([self.to_text(r) for r in rows], y, [r["group_id"] for r in rows])
        return self

    def predict_proba(self, rows):
        return self.search.predict_proba([self.to_text(r) for r in rows])[:, 1]

    def config(self):
        return {"C": self.search.best_params_["logisticregression__C"], "cv_auroc": self.search.best_score_}


def baselines():
    return {
        "majority": Majority(),
        "harmful_variant": HarmfulVariant(),
        **{name: Text(to_text) for name, to_text in TEXT_BASELINES.items()},
    }


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    precision, recall, _, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    two_classes = 0 < y.sum() < len(y)
    return {
        "n": len(y),
        "n_refuse": int(y.sum()),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred) if two_classes else None,
        "auroc": roc_auc_score(y, prob) if two_classes else None,
        # Rows are the judge label and columns the prediction, in the order COMPLY, REFUSE.
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "precision": {"COMPLY": precision[0], "REFUSE": precision[1]},
        "recall": {"COMPLY": recall[0], "REFUSE": recall[1]},
    }


def bootstrap(y, prob, groups, rng):
    """95% CIs for AUROC and balanced accuracy, resampling whole groups so twins stay together."""
    unique = sorted(set(groups))
    members = {g: [i for i, h in enumerate(groups) if h == g] for g in unique}
    pred = (prob >= 0.5).astype(int)
    samples = {"auroc": [], "balanced_accuracy": []}
    for _ in range(BOOTSTRAPS):
        idx = [i for g in rng.choice(unique, len(unique)) for i in members[g]]
        ys = y[idx]
        if not 0 < ys.sum() < len(ys):
            continue
        samples["auroc"].append(roc_auc_score(ys, prob[idx]))
        samples["balanced_accuracy"].append(balanced_accuracy_score(ys, pred[idx]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None for k, v in samples.items()}


def score(model, rows, y, rng):
    prob = model.predict_proba(rows)
    result = {}
    for name, keep in SUBSETS.items():
        mask = np.array([keep(r) for r in rows])
        sub = metrics(y[mask], prob[mask])
        sub["ci"] = bootstrap(y[mask], prob[mask], [r["group_id"] for r, m in zip(rows, mask) if m], rng)
        result[name] = sub
    return result


def run(rows, y, splits):
    """Fit every baseline on train and score it on each of splits."""
    train_rows, train_y = subset(rows, y, {"train"})
    rng = np.random.default_rng(SEED)
    out = {}
    for name, model in baselines().items():
        model.fit(train_rows, train_y)
        out[name] = {"config": model.config() if isinstance(model, Text) else {}}
        for split in splits:
            out[name][split] = score(model, *subset(rows, y, {split}), rng)
    return out


def fmt(value):
    return "  -  " if value is None else f"{value:.3f}"


def summarize(out, split):
    print(f"{split}: AUROC [95% CI] by subset, and balanced accuracy on all")
    print(f"{'baseline':16s} {'all':>20s} {'l2':>20s} {'l3':>20s} {'harmful':>20s} {'bal acc':>8s}")
    for name, result in out.items():
        cells = []
        for sub in ("all", "l2", "l3", "harmful"):
            r = result[split][sub]
            ci = r["ci"]["auroc"]
            cells.append(f"{fmt(r['auroc'])} [{fmt(ci[0])},{fmt(ci[1])}]" if ci else fmt(r["auroc"]))
        print(f"{name:16s} " + " ".join(f"{c:>20s}" for c in cells) + f" {fmt(result[split]['all']['balanced_accuracy']):>8s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="judge_labels.jsonl", help="file name inside results/")
    parser.add_argument("--final", action="store_true", help="score on test and heldout")
    args = parser.parse_args()

    rows, y = load(args.labels)
    splits = ["val", "test", "heldout"] if args.final else ["val"]
    out = run(rows, y, splits)
    output = RESULTS / ("baselines_final.json" if args.final else "baselines_val.json")
    output.write_text(json.dumps(out, indent=2) + "\n")
    for split in splits:
        summarize(out, split)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
