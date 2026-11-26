"""Assemble the local Stable Diffusion 2.1-base directory used by Kaggle."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

from huggingface_hub import snapshot_download

from .prepare_sd_tokenizer import REPO_ID, REVISION, prepare_tokenizer


COMPONENTS = {
    "text_encoder": ("model", "pytorch_model"),
    "unet": ("diffusion_pytorch_model", "diffusion_pytorch_model"),
    "vae": ("diffusion_pytorch_model", "diffusion_pytorch_model"),
}

SCHEDULER = {
    "_class_name": "PNDMScheduler", "beta_end": 0.012, "beta_schedule": "scaled_linear",
    "beta_start": 0.00085, "clip_sample": False, "num_train_timesteps": 1000,
    "prediction_type": "epsilon", "set_alpha_to_one": False, "skip_prk_steps": True,
    "steps_offset": 1, "trained_betas": None,
}


def component_files(directory, stems):
    config = directory / "config.json"
    if not config.is_file() or config.stat().st_size == 0:
        return None
    for variant in ("", ".fp16"):
        for stem, ext in zip(stems, ("safetensors", "bin")):
            name = f"{stem}.{ext}"
            weight = directory / f"{stem}{variant}.{ext}"
            if weight.is_file() and weight.stat().st_size > 1024:
                return {"config.json": config.resolve(), name: weight.resolve()}
    return None


def replace_component(destination, files):
    with TemporaryDirectory(prefix=".sd-component-", dir=destination.parent) as tmp:
        staged = Path(tmp) / destination.name
        staged.mkdir()
        backup = None
        if destination.exists() or destination.is_symlink():
            backup_dir = Path(mkdtemp(prefix=f".{destination.name}-original-", dir=destination.parent))
            backup = backup_dir / destination.name
        for name, source in files.items():
            if backup is not None and not destination.is_symlink() and source.is_relative_to(destination):
                source = backup / source.relative_to(destination)
            (staged / name).symlink_to(source)
        if backup is not None:
            destination.rename(backup)
        try:
            staged.rename(destination)
        except OSError:
            if backup is not None:
                backup.rename(destination)
            raise


def prepare_sd_base(source, sd_dir, allow_download=False):
    source, sd_dir = source.resolve(), sd_dir.resolve()
    search_dir = source.parent if source.name == "sd_v2_1" else source
    nested = sorted(search_dir.rglob("config.json")) if search_dir.is_dir() else []
    selected = {}
    for name, stems in COMPONENTS.items():
        candidates = [sd_dir / name, source / name]
        candidates.extend(p.parent for p in nested if p.parent.name == name)
        for directory in dict.fromkeys(candidates):
            files = component_files(directory, stems)
            if files is not None:
                selected[name] = (directory, files)
                break

    missing = [name for name in COMPONENTS if name not in selected]
    if missing:
        print(f"Missing SD 2.1 components: {', '.join(missing)}. Searched {sd_dir} and {search_dir}.", flush=True)
        if not allow_download:
            raise FileNotFoundError("Attach the SD 2.1-base dataset or pass --allow-download.")
        names = [
            file
            for name in missing
            for file in (f"{name}/config.json", f"{name}/{COMPONENTS[name][0]}.safetensors")
        ]
        print(f"Downloading missing components from {REPO_ID}: {', '.join(missing)}", flush=True)
        snapshot = Path(snapshot_download(
            repo_id=REPO_ID, revision=REVISION, allow_patterns=names, token=False,
        ))
        for name in missing:
            directory = snapshot / name
            files = component_files(directory, COMPONENTS[name])
            if files is None:
                raise FileNotFoundError(f"SD 2.1 component missing from the pinned snapshot: {directory}")
            selected[name] = (directory, files)

    sd_dir.mkdir(parents=True, exist_ok=True)
    for name, (directory, files) in selected.items():
        destination = sd_dir / name
        if directory != destination or any((destination / file).resolve() != path for file, path in files.items()):
            replace_component(destination, files)
        weights = ", ".join(str(path) for file, path in files.items() if file != "config.json")
        print(f"{name} OK: {weights}", flush=True)

    tokenizer_dirs = [source / "tokenizer"]
    tokenizer_dirs.extend(directory.parent / "tokenizer" for directory, _ in selected.values())
    tokenizer_source = next((directory for directory in tokenizer_dirs if directory.is_dir()), None)
    tokenizer = prepare_tokenizer(sd_dir, tokenizer_source)
    print(f"CLIP OK: {tokenizer.vocab_size} tokens, context length {tokenizer.model_max_length}")
    scheduler = sd_dir / "scheduler"
    if scheduler.is_symlink():
        scheduler.unlink()
    scheduler.mkdir(exist_ok=True)
    (scheduler / "scheduler_config.json").write_text(json.dumps(SCHEDULER, indent=2), encoding="utf-8")
    print(f"SD 2.1 base OK: {sd_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Path to the mounted SD 2.1-base dataset")
    parser.add_argument("--sd-dir", type=Path, required=True, help="Writable directory for the assembled model")
    parser.add_argument("--allow-download", action="store_true", help="Download missing model weights")
    args = parser.parse_args()
    prepare_sd_base(args.source, args.sd_dir, allow_download=args.allow_download)


if __name__ == "__main__":
    main()
