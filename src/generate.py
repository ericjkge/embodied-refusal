"""Run the target model on data/scenarios.jsonl and write the responses to results/."""
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
SCENARIOS = ROOT / "data" / "scenarios.jsonl"
RESULTS = ROOT / "results"
SYSTEM_PROMPT = (ROOT / "prompts" / "system_prompt.txt").read_text().strip()

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 256
SEED = 0


def build_prompt(tokenizer, row):
    user_prompt = f"Scene: {row['scene']}\nOperator instruction: {row['instruction']}"
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def sample_pairs(rows, pairs):
    """Keep a random set of scenario pairs."""
    scenario_ids = sorted({row["scenario_id"] for row in rows})
    keep = set(random.Random(SEED).sample(scenario_ids, pairs))
    return [row for row in rows if row["scenario_id"] in keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only run the first N scenario rows")
    parser.add_argument("--output", default="responses.jsonl", help="file name inside results/")
    parser.add_argument("--splits", help="comma-separated splits to run. Defaults to all")
    parser.add_argument("--pairs", type=int, help="randomly sample this many scenario pairs")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    eos_ids = set(model.generation_config.eos_token_id)

    rows = [json.loads(line) for line in SCENARIOS.read_text().splitlines()]
    if args.splits:
        rows = [row for row in rows if row["split"] in args.splits.split(",")]
    if args.pairs:
        rows = sample_pairs(rows, args.pairs)
    rows = rows[: args.limit]
    jobs = [(row, sample) for row in rows for sample in range(args.samples)]

    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / args.output
    with output.open("w") as f:
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start : start + args.batch_size]
            prompts = [build_prompt(tokenizer, row) for row, _ in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=1.0,
                    top_k=0,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
            new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
            for (row, sample), tokens in zip(batch, new_tokens):
                f.write(json.dumps({
                    **row,
                    "sample": sample,
                    "model": MODEL,
                    "revision": REVISION,
                    "temperature": TEMPERATURE,
                    "seed": SEED,
                    "finish_reason": "stop" if eos_ids & set(tokens.tolist()) else "length",
                    "response": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                }) + "\n")
            f.flush()
            print(f"{start + len(batch)}/{len(jobs)}", flush=True)
    print(f"Wrote {len(jobs)} responses to {output}")


if __name__ == "__main__":
    main()
