"""Baselines for the refusal probe: predict the judge label (REFUSE vs COMPLY) without activations.

- majority:       always predict the majority train class (COMPLY). AUROC is 0.5 by construction.
- harmful_rule:   predict REFUSE for every harmful variant and COMPLY for every benign one.
- tfidf_prompt:   TF-IDF + logistic regression on the prompt only (scene + instruction).
                  This is the fair comparison for the prompt_last probe, which also sees only the prompt.
- tfidf_response: TF-IDF + logistic regression on the full response. Expected to be near perfect,
                  because the response writes out the action. Sanity check only.

Uses the same protocol as probe.py: fit on train, choose C by 5-fold GroupKFold on group_id with
AUROC, and score on val. --final scores val, test, and heldout into results/baselines_final.json.
Run --final only once, together with `probe.py --final`, because test and heldout must not guide
development.

Needs only results/judge_labels.jsonl, so it runs on CPU without activations.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FOLDS = 5
# TF-IDF features are sparse and unstandardized, so they need a larger C range than the
# standardized activations in probe.py (1e-4 to 1).
CS = [0.01, 0.1, 1.0, 10.0, 100.0]
BASELINES = ("majority", "harmful_rule", "tfidf_prompt", "tfidf_response")


def load(labels_name):
    """Return judge-labeled rows with a REFUSE/COMPLY label, and y (1 = REFUSE)."""
    rows = [json.loads(line) for line in (RESULTS / labels_name).read_text().splitlines()]
    rows = [r for r in rows if r["label"] in ("REFUSE", "COMPLY")]
    y = np.array([r["label"] == "REFUSE" for r in rows], dtype=int)
    return rows, y


def prompt_text(row):
    return f"{row['scene']} {row['instruction']}"


def response_text(row):
    return row["response"]


def auroc(y, prob):
    return roc_auc_score(y, prob) if 0 < y.sum() < len(y) else None


def metrics(y, pred, prob):
    return {
        "n": len(y),
        "n_refuse": int(y.sum()),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "auroc": auroc(y, prob),
    }


def score(pred, prob, y, rows):
    """Score overall, per level, on levels 2 and 3 pooled, and within harmful variants at levels 2 and 3."""
    level = np.array([r["difficulty"] for r in rows])
    harmful = np.array([r["variant"] == "harmful" for r in rows])
    result = metrics(y, pred, prob)
    masks = {
        "l1": level == 1,
        "l2": level == 2,
        "l3": level == 3,
        "l23": level >= 2,
        "l23_harmful": (level >= 2) & harmful,
    }
    for name, mask in masks.items():
        for key, value in metrics(y[mask], pred[mask], prob[mask]).items():
            result[f"{key}_{name}"] = value
    return result


class Constant:
    """Majority-class baseline. Its constant score gives AUROC 0.5."""

    def fit(self, rows, y, groups):
        self.label = int(y.mean() > 0.5)
        return self

    def predict(self, rows):
        pred = np.full(len(rows), self.label)
        return pred, pred.astype(float)

    def params(self):
        return {"label": "REFUSE" if self.label else "COMPLY"}


class HarmfulRule:
    """Predict REFUSE exactly for the harmful variants. Needs no training."""

    def fit(self, rows, y, groups):
        return self

    def predict(self, rows):
        pred = np.array([r["variant"] == "harmful" for r in rows], dtype=int)
        return pred, pred.astype(float)

    def params(self):
        return {}


class Tfidf:
    """TF-IDF + L2 logistic regression, with C chosen by grouped cross-validation on train.

    The vectorizer sits inside the pipeline, so its vocabulary and IDF are refit on each fold.
    """

    def __init__(self, text):
        self.text = text

    def fit(self, rows, y, groups):
        folds = list(GroupKFold(n_splits=FOLDS).split(rows, y, groups))
        pipeline = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            LogisticRegression(max_iter=2000),
        )
        search = GridSearchCV(pipeline, {"logisticregression__C": CS}, cv=folds, scoring="roc_auc")
        self.model = search.fit([self.text(r) for r in rows], y)
        return self

    def predict(self, rows):
        texts = [self.text(r) for r in rows]
        return self.model.predict(texts), self.model.predict_proba(texts)[:, 1]

    def params(self):
        return {"C": self.model.best_params_["logisticregression__C"], "cv_auroc": self.model.best_score_}


def make(name):
    return {
        "majority": Constant,
        "harmful_rule": HarmfulRule,
        "tfidf_prompt": lambda: Tfidf(prompt_text),
        "tfidf_response": lambda: Tfidf(response_text),
    }[name]()


def subset(rows, y, splits):
    idx = [i for i, r in enumerate(rows) if r["split"] in splits]
    return [rows[i] for i in idx], y[idx]


def run(rows, y, splits):
    """Fit every baseline on train and score it on each split in splits."""
    train_rows, train_y = subset(rows, y, {"train"})
    groups = [r["group_id"] for r in train_rows]
    results = []
    for name in BASELINES:
        model = make(name).fit(train_rows, train_y, groups)
        out = {"baseline": name, **model.params()}
        for split in splits:
            split_rows, split_y = subset(rows, y, {split})
            pred, prob = model.predict(split_rows)
            out[split] = score(pred, prob, split_y, split_rows)
        results.append(out)
    return results


def fmt(value):
    return "  n/a" if value is None else f"{value:.3f}"


def show(results, split):
    print(f"\n{split} (AUROC / balanced accuracy)")
    print(f"{'baseline':16s} {'all':>13s} {'level 2':>13s} {'level 3':>13s} {'levels 2+3':>13s} {'l2+3 harmful':>13s}")
    for r in results:
        s = r[split]
        cells = [f"{fmt(s[f'auroc{k}'])}/{fmt(s[f'balanced_accuracy{k}'])}" for k in ("", "_l2", "_l3", "_l23", "_l23_harmful")]
        print(f"{r['baseline']:16s} " + " ".join(f"{c:>13s}" for c in cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="judge_labels.jsonl", help="file name inside results/")
    parser.add_argument("--output", default="baselines_val.json", help="file name inside results/")
    parser.add_argument("--final", action="store_true", help="also score on test and heldout")
    args = parser.parse_args()

    rows, y = load(args.labels)
    splits = ("val", "test", "heldout") if args.final else ("val",)
    output = RESULTS / ("baselines_final.json" if args.final else args.output)
    results = run(rows, y, splits)
    output.write_text(json.dumps(results, indent=2) + "\n")
    for split in splits:
        show(results, split)
    print(f"\nWrote {len(results)} baselines to {output}")


if __name__ == "__main__":
    main()
