"""Collect residual-stream activations for each generated response and save them to activations/.

Each response is re-run through the target model with teacher forcing (prompt + response).
Hidden states from every layer are kept at three token positions:
  prompt_last     the last prompt token, which predicts the first response token
  response_first  the first response token
  response_last   the last response token, before the end-of-turn token
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate import MODEL, RESULTS, REVISION, build_prompt

ROOT = Path(__file__).parent.parent
ACTIVATIONS = ROOT / "activations"
POSITIONS = ["prompt_last", "response_first", "response_last"]
META_KEYS = [
    "scenario_id", "variant", "sample", "hazard_type", "difficulty",
    "group_id", "split", "finish_reason",
]


def token_ids(tokenizer, row):
    """Tokenize the prompt and response separately so no token spans the boundary."""
    prompt_ids = tokenizer(build_prompt(tokenizer, row), add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
    if not response_ids:
        raise ValueError(f"empty response for {row['scenario_id']} {row['variant']}")
    return prompt_ids, response_ids


def extract(model, tokenizer, row):
    """Return a [positions, layers, hidden] tensor of activations for one response."""
    prompt_ids, response_ids = token_ids(tokenizer, row)
    ids = torch.tensor([prompt_ids + response_ids], device=model.device)
    indices = [len(prompt_ids) - 1, len(prompt_ids), ids.shape[1] - 1]
    with torch.no_grad():
        hidden_states = model(ids, output_hidden_states=True).hidden_states
    return torch.stack([h[0, indices] for h in hidden_states], dim=1).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="responses.jsonl", help="file name inside results/")
    parser.add_argument("--output", default="acts.pt", help="file name inside activations/")
    parser.add_argument("--limit", type=int, help="only run the first N responses")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    rows = [json.loads(line) for line in (RESULTS / args.input).read_text().splitlines()]
    rows = rows[: args.limit]
    acts = []
    for i, row in enumerate(rows):
        acts.append(extract(model, tokenizer, row))
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"{i + 1}/{len(rows)}", flush=True)

    ACTIVATIONS.mkdir(exist_ok=True)
    output = ACTIVATIONS / args.output
    torch.save({
        "model": MODEL,
        "revision": REVISION,
        "input": args.input,
        "positions": POSITIONS,
        "layers": list(range(acts[0].shape[1])),
        "acts": torch.stack(acts),
        "meta": [{key: row.get(key) for key in META_KEYS} for row in rows],
    }, output)
    print(f"Wrote activations {tuple(torch.stack(acts).shape)} for {len(rows)} responses to {output}")


if __name__ == "__main__":
    main()
