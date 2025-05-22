#!/usr/bin/env python

import json
from argparse import ArgumentParser
from pathlib import Path

from sardalign.utils import count_lines
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(description="Check the MLS dataset for consistency.")
    parser.add_argument("stok_mls_path", type=Path, help="Path to the MLS dataset.")
    return parser.parse_args()


def main(args):
    n_lines = count_lines(args.stok_mls_path)

    with open(args.stok_mls_path, "r") as f:
        for i, line in enumerate(tqdm(f, total=n_lines)):
            sample = json.loads(line)
            assert len(sample) == 1, f"Line {i} has more than one sample: {sample}"
            for j, (key, value) in enumerate(next(iter(sample.values())).items()):
                assert key == str(j), f"Line {i} has key {key} instead of {j}"
                assert isinstance(value, list), f"Line {i} has value {value} instead of a list"
                assert value, f"Line {i} has empty list for key {key}"
                assert all(isinstance(x, int) for x in value), f"Line {i} has non-int values in {value}"


if __name__ == "__main__":
    args = parse_args()
    main(args)
