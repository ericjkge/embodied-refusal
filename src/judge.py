"""Label responses in results/ with the LLM judge and write the labels to results/."""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
RUBRIC = (ROOT / "prompts" / "judge_rubric.txt").read_text()

MODEL = "claude-sonnet-5-5"
EFFORT = "low"
MAX_TOKENS = 2048
SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "label": {"type": "string", "enum": ["COMPLY", "REFUSE", "UNCLEAR"]},
    },
    "required": ["reasoning", "label"],
    "additionalProperties": False,
}


def refusal_type(label, response):
    """Read the kind of refusal from the action. It is for reporting only."""
    if label != "REFUSE":
        return None
    match = re.search(r"Action:\s*`?([a-z_]+)\(", response)
    action = match.group(1) if match else None
    return action if action in ("decline", "wait") else "other"


def judge(client, row):
    prompt = RUBRIC.format(
        scene=row["scene"], instruction=row["instruction"], response=row["response"]
    )
    message = client.beta.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    result = {"reasoning": None, "label": None}
    if message.stop_reason not in ("refusal", "max_tokens"):
        text = next(block.text for block in message.content if block.type == "text")
        result = json.loads(text)
    return {
        **row,
        "judge_model": message.model,
        "judge_stop_reason": message.stop_reason,
        "judge_reasoning": result["reasoning"],
        "label": result["label"],
        "refusal_type": refusal_type(result["label"], row["response"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        default="train,val",
        help="comma-separated splits to label. Add test,heldout only after freezing the rubric",
    )
    parser.add_argument("--limit", type=int, help="only label the first N responses")
    parser.add_argument("--input", default="responses.jsonl", help="file name inside results/")
    parser.add_argument("--output", default="judge_labels.jsonl", help="file name inside results/")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()

    splits = args.splits.split(",")
    rows = [json.loads(line) for line in (RESULTS / args.input).read_text().splitlines()]
    rows = [row for row in rows if row["split"] in splits][: args.limit]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda row: judge(client, row), rows))

    output = RESULTS / args.output
    with output.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Wrote {len(results)} labels to {output}")


if __name__ == "__main__":
    main()
