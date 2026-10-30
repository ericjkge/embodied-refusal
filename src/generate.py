"""Run the target model on data/scenarios.jsonl and write data/responses.jsonl."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).parent.parent
SCENARIOS = ROOT / "data" / "scenarios.jsonl"
RESPONSES = ROOT / "data" / "responses.jsonl"
SYSTEM_PROMPT = (ROOT / "prompts" / "system_prompt.txt").read_text().strip()

# Qwen3-8B isn't on OpenRouter, so the pilot uses the closest Qwen3 size.
MODEL = "qwen/qwen3-14b"
TEMPERATURE = 0.7
MAX_TOKENS = 256


def generate(client, model, row, sample):
    user_prompt = f"Scene: {row['scene']}\nOperator instruction: {row['instruction']}"
    completion = client.chat.completions.create(
        model=model,
        messages=[
            # OpenRouter ignores reasoning.enabled=False for Qwen3; /no_think works.
            {"role": "system", "content": SYSTEM_PROMPT + "\n/no_think"},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return {
        **row,
        "sample": sample,
        "model": model,
        "temperature": TEMPERATURE,
        "provider": getattr(completion, "provider", None),
        "finish_reason": completion.choices[0].finish_reason,
        "response": completion.choices[0].message.content,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--limit", type=int, help="only run the first N scenario rows")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_KEY"],
    )

    rows = [json.loads(line) for line in SCENARIOS.read_text().splitlines()]
    rows = rows[: args.limit]
    jobs = [(row, sample) for row in rows for sample in range(args.samples)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda job: generate(client, args.model, *job), jobs))

    with RESPONSES.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Wrote {len(results)} responses to {RESPONSES}")


if __name__ == "__main__":
    main()
