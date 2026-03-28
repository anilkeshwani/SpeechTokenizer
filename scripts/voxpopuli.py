#!/usr/bin/env python

"""
Encode the Voxpopuli (en subset) dataset using SpeechTokenizer.
Note: this script assumes
- standardised JSON lines input format
- local VoxPopuli dataset and filelists (see constants below)

Recall:
RVQ 0          -> Contains content info, can be considered as semantic tokens
RVQs 1 onwards -> Contain timbre info, complete info lost by the first quantizer
"""

import json
import os
import warnings
from argparse import ArgumentParser, Namespace
from math import ceil
from pathlib import Path
from typing import Any

import torch
import torchaudio
from speechtokenizer import SpeechTokenizer
from tqdm import tqdm


# Constants
VOXPOPULI_SPLIT_SIZES = {"train": 182_482, "validation": 1_753, "test": 1_842}
DEVICE = torch.device("cuda")  # leave GPU assignment to Slurm

# SpeechTokenizer Model
ST_MODEL_DIR = "/mnt/scratch-artemis/anilkeshwani/models/SpeechTokenizer/speechtokenizer_hubert_avg"
CONFIG_PATH = os.path.join(ST_MODEL_DIR, "config.json")
CKPT_PATH = os.path.join(ST_MODEL_DIR, "SpeechTokenizer.pt")

# Local VoxPopuli dataset manifest (filelist) paths
VOXPOPULI_FILELIST_DIR_DEFAULT = Path("/mnt/scratch-artemis/anilkeshwani/voxpopuli_filelists")
_VOXPOPULI_FILELIST_NAME_DEFAULT = "{}_uroman_aligned_hubert.jsonl" # format with split


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Encode VoxPopuli dataset with SpeechTokenizer.")
    # Required
    parser.add_argument("idx_block", type=int, help="Block index to process (0-based)")
    parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"])
    # Optional
    parser.add_argument("--block_size", type=int, default=100_000)  # 200k -> whole VP train split in 1 block
    parser.add_argument("--output_jsonl", type=Path)
    args = parser.parse_args()

    if args.idx_block < 0 or args.idx_block * args.block_size >= VOXPOPULI_SPLIT_SIZES[args.split]:
        raise ValueError(
            f"Invalid block index {args.idx_block} for split '{args.split}' and block size {args.block_size}."
        )

    return args


@torch.inference_mode()
def stok_encode_voxpopuli(idx_block: int, block_size: int, split: str, output_jsonl: Path | None):
    model = SpeechTokenizer.load_from_checkpoint(CONFIG_PATH, CKPT_PATH)
    model.eval()
    model.to(DEVICE)

    split_size = VOXPOPULI_SPLIT_SIZES[split]
    filelist = VOXPOPULI_FILELIST_DIR_DEFAULT / _VOXPOPULI_FILELIST_NAME_DEFAULT.format(split)
    n_blocks = ceil(split_size / block_size)

    # Get the block of MLS IDs to process
    start_idx = idx_block * block_size
    end_idx = min((idx_block + 1) * block_size, split_size)
    mls_ids = mls_ids[start_idx:end_idx]

    if output_jsonl is None:
        idx_block_label = str(idx_block + 1).zfill(len(str(n_blocks)))  # NOTE 1-indexed block label
        jsonl_filename = f"{split}-mls-speechtokenizer-{idx_block_label}-of-{n_blocks}.jsonl"
        output_jsonl = Path("/mnt/scratch-artemis/anilkeshwani/mls-speechtokenizer-jsonl") / split / jsonl_filename

    with open(output_jsonl, "x") as f:
        for mls_id in tqdm(mls_ids, desc="Processing MLS with SpeechTokenizer"):

            audio_path: str | Path  # TODO

            # Load and pre-process speech waveform
            wav, sr = torchaudio.load(audio_path)

            # monophonic checking
            if wav.size(0) > 1:
                warnings.warn(f"Audio {mls_id} is not monophonic. Shape: {wav.shape}. Taking the first channel.")
                wav = wav[:1, :]

            # sample rate checking
            if sr != model.sample_rate:
                warnings.warn(f"Audio {mls_id} has sample rate {sr} != {model.sample_rate}. Resampling.")
                wav = torchaudio.functional.resample(wav, sr, model.sample_rate)

            # Extract discrete codes from SpeechTokenizer
            codes = model.encode(wav.to(DEVICE).unsqueeze(0)).squeeze(1)  # codes: (n_q, T)

            # Write RVQ codes to file
            stok_rvqs: dict[str, list[int]] = {f"RVQ_{idx_q}": st.tolist() for idx_q, st in enumerate(codes)}
            stok_sample: dict[str, Any] = {"ID": mls_id} | stok_rvqs

            f.write(json.dumps(stok_sample) + "\n")

    print(f"Completed. Encoded block {idx_block} to {output_jsonl}.")


if __name__ == "__main__":
    stok_encode_mls(**vars(parse_args()))
