#!/usr/bin/env python

"""Validate SpeechTokenizer-encoded MLS JSONL files.

Expected format per line (flat JSON, matching mls.py output):
    {"ID": "<MLS_ID>", "RVQ_0": [int, ...], "RVQ_1": [int, ...], ..., "RVQ_{n_q-1}": [int, ...]}
"""

import json
from argparse import ArgumentParser
from pathlib import Path

from sardalign.utils import count_lines
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(description="Validate SpeechTokenizer-encoded MLS JSONL files.")
    parser.add_argument("stok_mls_path", type=Path, help="Path to the JSONL file to validate.")
    parser.add_argument("--n_q", type=int, default=8, help="Expected number of RVQ layers (default: 8).")
    return parser.parse_args()


def main(args):
    n_lines = count_lines(args.stok_mls_path)
    expected_keys = {"ID"} | {f"RVQ_{i}" for i in range(args.n_q)}

    with open(args.stok_mls_path, "r") as f:
        for i, line in enumerate(tqdm(f, total=n_lines)):
            sample = json.loads(line)

            assert set(sample.keys()) == expected_keys, (
                f"Line {i}: expected keys {sorted(expected_keys)} but got {sorted(sample.keys())}"
            )

            assert isinstance(sample["ID"], str) and sample["ID"], (
                f"Line {i}: 'ID' must be a non-empty string, got {sample['ID']!r}"
            )

            for q in range(args.n_q):
                key = f"RVQ_{q}"
                value = sample[key]
                assert isinstance(value, list) and value, (
                    f"Line {i}: '{key}' must be a non-empty list, got {type(value).__name__}"
                )
                assert all(isinstance(x, int) for x in value), (
                    f"Line {i}: '{key}' contains non-int values"
                )

    print(f"Validated {n_lines} lines in {args.stok_mls_path} — all OK.")


if __name__ == "__main__":
    main(parse_args())
