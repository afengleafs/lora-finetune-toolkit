#!/usr/bin/env python3
"""
Convert JSONL input/output pairs to the JSON list format used for training.

Usage:
    python scripts/convert_jsonl_to_training_json.py \
        --input data/examples/qwen3_finetune_paper_lab_500.jsonl \
        --output data/examples/qwen3_finetune_paper_lab_500.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSONL training data to a JSON list of input/output pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data/examples/qwen3_finetune_paper_lab_500.jsonl",
        help="Path to the source JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="data/examples/qwen3_finetune_paper_lab_500.json",
        help="Path to write the converted JSON file.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Use 0 for compact output.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc}") from exc

            missing = [key for key in ("input", "output") if key not in item]
            if missing:
                raise ValueError(
                    f"Line {line_number}: missing required field(s): {', '.join(missing)}"
                )

            input_text = str(item["input"]).strip()
            output_text = str(item["output"]).strip()
            if not input_text or not output_text:
                raise ValueError(f"Line {line_number}: input/output cannot be empty")

            rows.append({"input": input_text, "output": output_text})

    return rows


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    rows = load_jsonl(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    indent = None if args.indent == 0 else args.indent
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=indent)
        f.write("\n")

    print(f"Converted {len(rows)} examples")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
