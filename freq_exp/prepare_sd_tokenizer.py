"""Prepare and verify the Stable Diffusion 2.1-base tokenizer for Kaggle."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

from huggingface_hub import snapshot_download
from transformers import CLIPTokenizerFast


REPO_ID = "sd-research/stable-diffusion-2-1-base"
REVISION = "0708cecd370b4d1c3a6ff3f7332f5e9aea78896f"
TOKENIZER_FILES = [
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
]


def load_tokenizer(path, **kwargs):
    tokenizer = CLIPTokenizerFast.from_pretrained(str(path), local_files_only=True, **kwargs)
    if (
        tokenizer.vocab_size != 49408
        or tokenizer.model_max_length != 77
        or tokenizer.bos_token_id != 49406
        or tokenizer.eos_token_id != 49407
        or tokenizer.pad_token_id is None
    ):
        raise ValueError(
            f"{path}: expected the SD 2.1 CLIP tokenizer, got "
            f"{tokenizer.vocab_size} tokens and context length {tokenizer.model_max_length}"
        )
    return tokenizer


def prepare_tokenizer(sd_dir, source):
    destination = sd_dir / "tokenizer"
    if destination.is_dir():
        try:
            return load_tokenizer(sd_dir, subfolder="tokenizer")
        except (OSError, ValueError, TypeError) as exc:
            print(f"Tokenizer at {destination} failed validation: {exc}", flush=True)
    else:
        print(f"Tokenizer directory missing or link broken: {destination}", flush=True)

    tokenizer = None
    for candidate in dict.fromkeys((destination, source)):
        if candidate is not None and candidate.is_dir():
            try:
                tokenizer = load_tokenizer(candidate)
            except (OSError, ValueError, TypeError) as exc:
                print(f"Cannot read tokenizer directly from {candidate}: {exc}", flush=True)
            else:
                break
    if tokenizer is None:
        print(f"Downloading SD 2.1 tokenizer files from {REPO_ID}", flush=True)
        snapshot = snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            allow_patterns=TOKENIZER_FILES,
            token=False,
        )
        tokenizer = load_tokenizer(Path(snapshot) / "tokenizer")

    sd_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".tokenizer-build-", dir=sd_dir) as tmp:
        staging = Path(tmp)
        tokenizer.save_pretrained(staging / "tokenizer")
        prepared = load_tokenizer(staging, subfolder="tokenizer")
        if prepared.pad_token_id != tokenizer.pad_token_id:
            raise ValueError("CLIP padding token changed during serialization")
        if destination.exists() or destination.is_symlink():
            backup = Path(mkdtemp(prefix=".tokenizer-original-", dir=sd_dir)) / "tokenizer"
            destination.rename(backup)
            print(f"Previous tokenizer kept at {backup}", flush=True)
        (staging / "tokenizer").rename(destination)
    return load_tokenizer(sd_dir, subfolder="tokenizer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sd-dir", required=True, type=Path)
    parser.add_argument("--source", type=Path, help="Tokenizer directory in the input dataset")
    args = parser.parse_args()
    tokenizer = prepare_tokenizer(args.sd_dir, args.source)
    print(f"CLIP OK: {tokenizer.vocab_size} tokens, context length {tokenizer.model_max_length}")


if __name__ == "__main__":
    main()
