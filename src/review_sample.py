"""Draw the manual-review sample of held-out responses (test + heldout) for Part 2 §4.

Takes every response where the probe and the judge disagree, plus an equal number of
agreements, drawn at random (seed 0) half from judge REFUSE and half from judge COMPLY,
preferring levels 2 and 3. Uses results/predictions_final.jsonl from src/predictions.py.

Writes:
  results/review_sample.jsonl  only what the labeling UI shows (scene, instruction, response)
                               plus ids, so the reviewer stays blind. Label it with
                               `python labeling/server.py --input review_sample.jsonl --output review_labels.jsonl`
  results/review_key.jsonl     judge label, probe prediction, and why each row was picked.
                               Don't open it until labeling is finished.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
SEED = 0
SPLITS = ("test", "heldout")
KEY = ("scenario_id", "variant", "sample")


def main():
    rng = random.Random(SEED)
    preds = [json.loads(line) for line in (RESULTS / "predictions_final.jsonl").read_text().splitlines()]
    preds = [p for p in preds if p["split"] in SPLITS]
    texts = {tuple(r[k] for k in KEY): r
             for r in map(json.loads, (RESULTS / "judge_labels.jsonl").read_text().splitlines())}

    judge = lambda p: int(p["label"] == "REFUSE")
    disagree = [p for p in preds if p["probe_pred"] != judge(p)]
    agree = [p for p in preds if p["probe_pred"] == judge(p)]

    picked = [(p, "disagree") for p in disagree]
    for label in (1, 0):
        pool = [p for p in agree if judge(p) == label]
        hard = [p for p in pool if p["difficulty"] >= 2]
        easy = [p for p in pool if p["difficulty"] == 1]
        rng.shuffle(hard)
        rng.shuffle(easy)
        want = (len(disagree) + (label == 1)) // 2
        picked += [(p, "agree") for p in (hard + easy)[:want]]

    rng.shuffle(picked)
    with (RESULTS / "review_sample.jsonl").open("w") as f_sample, (RESULTS / "review_key.jsonl").open("w") as f_key:
        for p, reason in picked:
            t = texts[tuple(p[k] for k in KEY)]
            f_sample.write(json.dumps({k: t[k] for k in KEY + ("split", "scene", "instruction", "response")}) + "\n")
            f_key.write(json.dumps({
                **{k: p[k] for k in KEY + ("split", "hazard_type", "difficulty")},
                "picked_as": reason,
                "judge_label": p["label"],
                "probe_pred": "REFUSE" if p["probe_pred"] else "COMPLY",
                "probe_prob": round(p["probe_prob"], 3),
                "tfidf_prompt_pred": "REFUSE" if p["tfidf_prompt_pred"] else "COMPLY",
                "judge_reasoning": t["judge_reasoning"],
            }) + "\n")
    n_agree = sum(r == "agree" for _, r in picked)
    print(f"Wrote {len(picked)} rows ({len(disagree)} disagreements, {n_agree} agreements) "
          "to results/review_sample.jsonl and results/review_key.jsonl")


if __name__ == "__main__":
    main()
