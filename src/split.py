"""Add group_id and split to each row of data/scenarios.jsonl.

Electrical is held out for the generalization check. The other hazards are
split 70/15/15 per hazard and level. Twins and related scenarios (scenes with
at least 40% word overlap) always land in the same split.

Rerun this after make_scenarios.py, which overwrites data/scenarios.jsonl.
"""
import json
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

SCENARIOS = Path(__file__).parent.parent / "data" / "scenarios.jsonl"

SEED = 0
HOLDOUT_HAZARD = "electrical"
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SIMILARITY = 0.4
STOPWORDS = set(
    "a an the of in on at to and is are with for it its his her their "
    "robot holds sits stands nobody near next".split()
)


def words(text):
    return set(re.findall(r"[a-z0-9-]+", text.lower())) - STOPWORDS


def group_scenarios(scenes):
    """Connect scenarios whose scenes overlap, and return scenario_id -> group_id."""
    parent = {scenario_id: scenario_id for scenario_id in scenes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in combinations(scenes, 2):
        overlap = len(scenes[a] & scenes[b]) / len(scenes[a] | scenes[b])
        if overlap >= SIMILARITY:
            parent[find(a)] = find(b)
    return {scenario_id: find(scenario_id) for scenario_id in scenes}


def assign_splits(groups, cell_of):
    """Give each group to the split that most needs its hazard and level."""
    cell_sizes = Counter(cell_of.values())
    counts = {split: Counter() for split in RATIOS}
    assignment = {}
    group_ids = sorted(groups)
    random.Random(SEED).shuffle(group_ids)
    group_ids.sort(key=lambda g: -len(groups[g]))
    for group_id in group_ids:
        cells = [cell_of[s] for s in groups[group_id]]
        split = max(
            RATIOS,
            key=lambda s: sum(RATIOS[s] * cell_sizes[c] - counts[s][c] for c in cells),
        )
        assignment[group_id] = split
        counts[split].update(cells)
    return assignment


def main():
    rows = [json.loads(line) for line in SCENARIOS.read_text().splitlines()]

    scenes = defaultdict(set)
    for row in rows:
        scenes[row["scenario_id"]] |= words(row["scene"])
    group_of = group_scenarios(scenes)

    cell_of = {
        row["scenario_id"]: (row["hazard_type"], row["difficulty"])
        for row in rows
        if row["hazard_type"] != HOLDOUT_HAZARD
    }
    groups = defaultdict(list)
    for scenario_id in cell_of:
        groups[group_of[scenario_id]].append(scenario_id)
    assignment = assign_splits(groups, cell_of)

    with SCENARIOS.open("w") as f:
        for row in rows:
            group_id = group_of[row["scenario_id"]]
            row["group_id"] = group_id
            row["split"] = assignment.get(group_id, "heldout")
            f.write(json.dumps(row) + "\n")

    summary = Counter((r["split"], r["difficulty"]) for r in rows if r["variant"] == "harmful")
    for split in [*RATIOS, "heldout"]:
        print(split, {level: summary[(split, level)] for level in (1, 2, 3)})


if __name__ == "__main__":
    main()
