#!/usr/bin/env python

"""
Encode the Multilingual LibriSpeech (MLS) dataset using SpeechTokenizer.

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
MLS_SIZES = {"train": 10_808_037, "dev": 3_807, "test": 3_769}
DEVICE = torch.device("cuda")  # leave GPU assignment to Slurm

# SpeechTokenizer Model
ST_MODEL_DIR = "/mnt/scratch-artemis/anilkeshwani/models/SpeechTokenizer/speechtokenizer_hubert_avg"
CONFIG_PATH = os.path.join(ST_MODEL_DIR, "config.json")
CKPT_PATH = os.path.join(ST_MODEL_DIR, "SpeechTokenizer.pt")

# Local MLS Dataset Paths
_MLS_SEGMENTS_PATH = "/mnt/scratch-artemis/shared/datasets/MLS/{}/segments.txt"
_MLS_AUDIO_DIR = "/mnt/scratch-artemis/shared/datasets/MLS/{}/audio"


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Encode MLS dataset with SpeechTokenizer.")
    # Required
    parser.add_argument("idx_block", type=int, help="Block index to process (0-based)")
    parser.add_argument("--split", type=str, required=True, choices=["train", "dev", "test"])
    # Optional
    parser.add_argument("--block_size", type=int, default=100_000)  # ~3:20 (3.3 hours) based on 500 samples/min tested
    parser.add_argument("--output_jsonl", type=Path)
    args = parser.parse_args()

    if args.idx_block < 0 or args.idx_block * args.block_size >= MLS_SIZES[args.split]:
        raise ValueError(
            f"Invalid block index {args.idx_block} for split '{args.split}' and block size {args.block_size}."
        )

    return args


def mls_id_to_path(mls_id: str, audio_dir: Path, suffix: str = ".flac") -> Path:
    """Infer path of the audio file from the MLS ID and audio directory.

    Args:
        mls_id (str): ID as found in transcripts.txt file e.g. 10214_10108_000000
        audio_dir (Path): "audio" directory e.g. /mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/dev/audio
        suffix (str, optional): File extension. Defaults to ".flac".

    Returns:
        Path: Resolved path pointing to audio file
    """
    speaker_id, book_id, file_specifier = mls_id.removesuffix(suffix).split("_")
    return (audio_dir / speaker_id / book_id / mls_id).with_suffix(suffix)


@torch.inference_mode()
def stok_encode_mls(idx_block: int, block_size: int, split: str, output_jsonl: Path | None):
    model = SpeechTokenizer.load_from_checkpoint(CONFIG_PATH, CKPT_PATH)
    model.eval()
    model.to(DEVICE)

    mls_split_size = MLS_SIZES[split]
    mls_segments = _MLS_SEGMENTS_PATH.format(split)
    mls_audio_dir = Path(_MLS_AUDIO_DIR.format(split))
    n_blocks = ceil(mls_split_size / block_size)

    with open(mls_segments, "r") as f:
        mls_ids: list[str] = [line.strip().split(None, 1)[0] for line in f]

    if len(mls_ids) != mls_split_size:
        raise ValueError(f"Expected {mls_split_size} MLS IDs in {mls_segments}, but found {len(mls_ids)}.")

    # Get the block of MLS IDs to process
    start_idx = idx_block * block_size
    end_idx = min((idx_block + 1) * block_size, mls_split_size)
    mls_ids = mls_ids[start_idx:end_idx]

    if output_jsonl is None:
        idx_block_label = str(idx_block + 1).zfill(len(str(n_blocks)))  # NOTE 1-indexed block label
        jsonl_filename = f"{split}-mls-speechtokenizer-{idx_block_label}-of-{n_blocks}.jsonl"
        output_jsonl = Path("/mnt/scratch-artemis/anilkeshwani/mls-speechtokenizer-jsonl") / split / jsonl_filename

    with open(output_jsonl, "x") as f:
        for mls_id in tqdm(mls_ids, desc="Processing MLS with SpeechTokenizer"):
            # Load and pre-process speech waveform
            wav, sr = torchaudio.load(mls_id_to_path(mls_id, mls_audio_dir))

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
