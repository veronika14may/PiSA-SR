"""Загрузка VAE и чекпоинтов дообученного декодера (К6).

Чекпоинт декодера — обычный словарь: {"state_dict": веса decoder.* и post_quant_conv.*, "meta": {...}}.
Энкодер в него не входит: он заморожен, поэтому латентное пространство остаётся тем же, и UNet, LoRA
и учитель CSD с новым декодером совместимы без изменений.
"""
import torch

DEC_PREFIXES = ("decoder.", "post_quant_conv.")


def load_vae(sd_path):
    """VAE из Stable Diffusion. Сначала копия класса из репозитория PiSA-SR; если она несовместима
    с установленной версией diffusers — стандартный AutoencoderKL (веса и имена параметров те же)."""
    try:
        from src.models.autoencoder_kl import AutoencoderKL
    except Exception as e:
        print(f"VAE из репозитория не загрузился ({type(e).__name__}: {e}); беру AutoencoderKL из diffusers, веса те же.")
        from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(sd_path, subfolder="vae")


def decoder_params(vae):
    """Обучаемая часть в К6: декодер и свёртка 1×1 перед ним."""
    return list(vae.decoder.parameters()) + list(vae.post_quant_conv.parameters())


def save_decoder(vae, path, meta=None):
    sd = {k: v.detach().float().cpu() for k, v in vae.state_dict().items() if k.startswith(DEC_PREFIXES)}
    torch.save({"state_dict": sd, "meta": meta or {}}, path)


def load_decoder(vae, path):
    """Подставляет в VAE веса дообученного декодера. Возвращает meta из чекпоинта."""
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("state_dict", ck)
    want = {k for k in vae.state_dict() if k.startswith(DEC_PREFIXES)}
    got = set(sd)
    assert got <= want and want <= got, (
        f"чекпоинт декодера не подходит к этому VAE: не хватает {sorted(want - got)[:3]}, лишние {sorted(got - want)[:3]}")
    vae.load_state_dict(sd, strict=False)
    print(f"[VAE] декодер заменён на дообученный: {path}")
    return ck.get("meta", {}) if isinstance(ck, dict) else {}
