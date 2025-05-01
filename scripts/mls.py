#!/usr/bin/env python

import json
import os
import sys
import warnings
from pathlib import Path

import torch
import torchaudio
from speechtokenizer import SpeechTokenizer
from tqdm import tqdm


# NOTE
# RVQ 0: Contains content info, can be considered as semantic tokens
# RVQs 1 onwards: Contain timbre info, complete info lost by the first quantizer

BLOCK_SIZE = 100_000  # ~3:20 (3.3 hours) based on 500 samples/minute from testing
# BLOCK_SIZE = 50  # testing
MLS_TRAIN_SIZE = 10_808_037  # number of samples in MLS train set

ST_MODEL_DIR = "/mnt/scratch-artemis/anilkeshwani/models/SpeechTokenizer/speechtokenizer_hubert_avg"
CONFIG_PATH = os.path.join(ST_MODEL_DIR, "config.json")
CKPT_PATH = os.path.join(ST_MODEL_DIR, "SpeechTokenizer.pt")
MLS_TRAIN_SEGMENTS = "/mnt/scratch-artemis/shared/datasets/MLS/train/segments.txt"
MLS_TRAIN_AUDIO_DIR = Path("/mnt/scratch-artemis/shared/datasets/MLS/train/audio")
OUTPUT_DIR = Path("/mnt/scratch-artemis/anilkeshwani/stok-mls")  # must exist

DEVICE = torch.device("cuda")  # leave GPU assignment to Slurm


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
def stok_encode_mls(idx_block: int, out_file: Path):
    model = SpeechTokenizer.load_from_checkpoint(CONFIG_PATH, CKPT_PATH)
    model.eval()
    model.to(DEVICE)

    mls_ids: list[str] = []
    with open(MLS_TRAIN_SEGMENTS, "r") as f:
        for i, line in enumerate(f):
            mls_ids.append(line.strip().split(None, 1)[0])

    assert len(mls_ids) == MLS_TRAIN_SIZE

    # Get the block of MLS IDs to process
    start_idx = idx_block * BLOCK_SIZE
    end_idx = min((idx_block + 1) * BLOCK_SIZE, MLS_TRAIN_SIZE)
    mls_ids = mls_ids[start_idx:end_idx]

    with open(out_file, "x") as f:
        for mls_id in tqdm(mls_ids, desc="Processing MLS with SpeechTokenizer"):
            # Load and pre-process speech waveform
            wav, sr = torchaudio.load(mls_id_to_path(mls_id, MLS_TRAIN_AUDIO_DIR))

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
            stok_rvqs: dict[int, list[int]] = {idx_q: st.tolist() for idx_q, st in enumerate(codes)}
            f.write(json.dumps({mls_id: stok_rvqs}) + "\n")


def main():
    idx_block = int(sys.argv[1])  # start from 0

    # Check block index is valid
    if idx_block < 0 or idx_block * BLOCK_SIZE >= MLS_TRAIN_SIZE:
        raise ValueError(f"Invalid block index {idx_block}. Must be between 0 and {MLS_TRAIN_SIZE // BLOCK_SIZE}.")

    out_file = OUTPUT_DIR / f"stok_mls_{idx_block!s}.jsonl"

    stok_encode_mls(idx_block, out_file)

    print(f"Completed. Encoded block {idx_block} to {out_file}.")


if __name__ == "__main__":
    main()
