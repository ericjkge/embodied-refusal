"""Local UI for hand-labeling responses. Saves after every label, so it can resume.

Run from the repo root:
    python labeling/server.py --input hand_label_sample.jsonl --output hand_labels.jsonl
"""
import argparse
import json
import os
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
PAGE = Path(__file__).parent / "index.html"

SEED = 0
LABELS = {"COMPLY", "REFUSE", "UNCLEAR"}
lock = threading.Lock()


def item_id(row):
    return f"{row['scenario_id']}:{row['variant']}:{row['sample']}"


class Store:
    def __init__(self, input_path, output_path):
        self.output_path = output_path
        self.rows = [json.loads(line) for line in input_path.read_text().splitlines()]
        # Shuffled so twins and levels aren't next to each other.
        random.Random(SEED).shuffle(self.rows)
        self.labels = {}
        if output_path.exists():
            for line in output_path.read_text().splitlines():
                saved = json.loads(line)
                self.labels[item_id(saved)] = saved

    def items(self):
        # Hazard, level, and variant stay hidden so they can't bias the label.
        return [
            {
                "id": item_id(row),
                "scene": row["scene"],
                "instruction": row["instruction"],
                "response": row["response"],
                "label": self.labels.get(item_id(row), {}).get("label"),
                "note": self.labels.get(item_id(row), {}).get("note", ""),
            }
            for row in self.rows
        ]

    def save(self, id_, label, note):
        row = next(r for r in self.rows if item_id(r) == id_)
        self.labels[id_] = {
            "scenario_id": row["scenario_id"],
            "variant": row["variant"],
            "sample": row["sample"],
            "split": row["split"],
            "label": label,
            "note": note,
        }
        temp_path = self.output_path.with_suffix(".tmp")
        with temp_path.open("w") as f:
            for saved in self.labels.values():
                f.write(json.dumps(saved) + "\n")
        os.replace(temp_path, self.output_path)


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def send(self, body, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/items":
                self.send(json.dumps(store.items()).encode(), "application/json")
            elif self.path == "/":
                self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/api/label":
                self.send_error(404)
                return
            length = int(self.headers["Content-Length"])
            data = json.loads(self.rfile.read(length))
            if data["label"] not in LABELS:
                self.send_error(400)
                return
            with lock:
                store.save(data["id"], data["label"], data.get("note", ""))
            self.send(b"{}", "application/json")

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="hand_label_sample.jsonl", help="file name inside results/")
    parser.add_argument("--output", default="hand_labels.jsonl", help="file name inside results/")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    store = Store(RESULTS / args.input, RESULTS / args.output)
    print(f"Open http://localhost:{args.port}. Saving to {store.output_path}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store)).serve_forever()


if __name__ == "__main__":
    main()
