"""Verify SpeechTokenizer encoding reproducibility against published MLS tokens.

Encodes the first N samples from each MLS English split with SpeechTokenizer and
asserts exact token-level match against the reference dataset on HuggingFace:
    https://huggingface.co/datasets/anilkeshwani/mls-speechtokenizer

Requires: GPU, internet access, ~2 GB disk for model + audio downloads.

Usage:
    pytest tests/test_mls_tokenization_reproducibility.py -v
    pytest tests/test_mls_tokenization_reproducibility.py -v -k "dev"     # single split
    pytest tests/test_mls_tokenization_reproducibility.py -v --n-samples 10  # fewer samples
"""

import pytest
import torch
import torchaudio
from datasets import load_dataset
from huggingface_hub import snapshot_download
from speechtokenizer import SpeechTokenizer


# ── Constants ────────────────────────────────────────────────────────────────

REFERENCE_DATASET = "anilkeshwani/mls-speechtokenizer"
MLS_DATASET = "facebook/multilingual_librispeech"
MLS_LANG = "english"
STOK_HF_REPO = "fnlp/SpeechTokenizer"
STOK_HF_SUBDIR = "speechtokenizer_hubert_avg"
N_Q = 8  # number of RVQ quantizers

# MLS split name -> reference dataset split name
SPLIT_MAPPING = {
    "train": "train",
    "dev": "validation",
    "test": "test",
}


# ── Fixtures ─────────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--n-samples", type=int, default=100, help="Number of samples to verify per split"
    )


@pytest.fixture(scope="session")
def n_samples(request):
    return request.config.getoption("--n-samples")


@pytest.fixture(scope="session")
def stok_model():
    """Download and load the SpeechTokenizer model once per test session."""
    model_dir = snapshot_download(repo_id=STOK_HF_REPO, allow_patterns=f"{STOK_HF_SUBDIR}/*")
    config_path = f"{model_dir}/{STOK_HF_SUBDIR}/config.json"
    ckpt_path = f"{model_dir}/{STOK_HF_SUBDIR}/SpeechTokenizer.pt"
    model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_reference_samples(split: str, n: int) -> dict[str, dict[str, list[int]]]:
    """Load first n samples from the reference dataset, keyed by ID."""
    ref_ds = load_dataset(REFERENCE_DATASET, split=f"{split}[:{n}]")
    ref_by_id = {}
    for row in ref_ds:
        sample_id = row["ID"]
        ref_by_id[sample_id] = {f"RVQ_{q}": row[f"RVQ_{q}"] for q in range(N_Q)}
    return ref_by_id


def load_mls_audio_by_ids(
    mls_split: str, target_ids: set[str], sample_rate: int
) -> dict[str, torch.Tensor]:
    """Stream MLS English and collect audio tensors for the target IDs."""
    ds = load_dataset(MLS_DATASET, MLS_LANG, split=mls_split, streaming=True)
    audio_by_id = {}
    for sample in ds:
        if sample["id"] in target_ids:
            wav = torch.tensor(sample["audio"]["array"], dtype=torch.float32)
            sr = sample["audio"]["sampling_rate"]
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            # Ensure mono (1, T) then add batch dim -> (1, 1, T)
            audio_by_id[sample["id"]] = wav.unsqueeze(0).unsqueeze(0)
            if len(audio_by_id) == len(target_ids):
                break  # found all targets, stop streaming
    return audio_by_id


@torch.inference_mode()
def encode_and_compare(
    model: SpeechTokenizer,
    audio_by_id: dict[str, torch.Tensor],
    ref_by_id: dict[str, dict[str, list[int]]],
) -> list[str]:
    """Encode each audio sample and compare against reference tokens. Returns list of failures."""
    device = next(model.parameters()).device
    failures = []
    for sample_id, wav in sorted(audio_by_id.items()):
        codes = model.encode(wav.to(device))  # (n_q, 1, T)
        codes = codes.squeeze(1).cpu()  # (n_q, T)
        ref = ref_by_id[sample_id]
        for q in range(N_Q):
            ref_tokens = ref[f"RVQ_{q}"]
            enc_tokens = codes[q].tolist()
            if enc_tokens != ref_tokens:
                failures.append(
                    f"ID={sample_id} RVQ_{q}: "
                    f"length {len(enc_tokens)} vs ref {len(ref_tokens)}, "
                    f"first mismatch at index "
                    f"{next((i for i, (a, b) in enumerate(zip(enc_tokens, ref_tokens)) if a != b), 'length_diff')}"
                )
    return failures


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mls_split,ref_split", list(SPLIT_MAPPING.items()))
def test_mls_tokenization_matches_reference(stok_model, n_samples, mls_split, ref_split):
    """Verify SpeechTokenizer produces identical tokens to the published reference."""
    ref_by_id = load_reference_samples(ref_split, n_samples)
    assert ref_by_id, f"No reference samples loaded for split '{ref_split}'"

    audio_by_id = load_mls_audio_by_ids(mls_split, set(ref_by_id.keys()), stok_model.sample_rate)

    missing = set(ref_by_id.keys()) - set(audio_by_id.keys())
    assert not missing, f"Could not find {len(missing)} reference IDs in MLS {mls_split}: {sorted(missing)[:5]}..."

    failures = encode_and_compare(stok_model, audio_by_id, ref_by_id)
    assert not failures, (
        f"{len(failures)} token mismatches in {mls_split} split:\n" + "\n".join(failures[:10])
    )
