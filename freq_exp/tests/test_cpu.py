"""Проверка потерь и метрик на CPU: значения, конечность градиентов, нормы градиентов.

Запуск из PiSA-SR-main:  python freq_exp/tests/test_cpu.py <папка HQ> [число снимков]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
import losses as L, metrics as M

d = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
names = sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))[:n]
hq = torch.stack([torch.from_numpy(np.array(Image.open(os.path.join(d, f)).convert("RGB"))).permute(2, 0, 1).float() / 127.5 - 1 for f in names])
lq = F.interpolate(F.interpolate(hq, scale_factor=0.25, mode="bicubic", antialias=True), size=hq.shape[-2:], mode="bicubic").clamp(-1, 1)
print("снимков:", len(names), "размер:", tuple(hq.shape))

ffl = L.build_ffl()
fdl = L.build_fdl(device="cpu", num_proj=64)
try:
    import lpips
    lp = lpips.LPIPS(net="vgg", spatial=True).requires_grad_(False)
except Exception as e:
    lp = None; print("lpips недоступен:", e)

C = 256
terms = {
    "L2": lambda p: F.mse_loss(p, hq),
    "L2 + карта деталей": lambda p: (L.daw_weight(hq, 1.0) * (p - hq) ** 2).mean(),
    "FFL": lambda p: ffl(p, hq),
    "Fourier Space Loss": lambda p: L.fourier_space_loss(p, hq),
    "FDL (кроп 256)": lambda p: L.fdl_call(fdl, p, hq, "full", crop=C),
    "FDL по областям": lambda p: L.fdl_call(fdl, p, hq, "full", region=L.daw_weight(hq, 1.0), crop=C),
    "FDL по границам": lambda p: L.fdl_call(fdl, p, hq, "edge", crop=C),
    "FDL по полосам": lambda p: L.fdl_call(fdl, p, hq, "bands", crop=C),
}
if lp is not None:
    terms["LPIPS"] = lambda p: lp(p, hq).mean()
    terms["LPIPS + карта деталей"] = lambda p: (L.daw_weight(hq, 1.0) * lp(p, hq)).mean()

print(f"{'член':24s} {'значение':>12} {'на эталоне':>12} {'норма град.':>12} {'конечен':>8}")
for k, fn in terms.items():
    p = lq.clone().requires_grad_(True)
    v = fn(p); g, = torch.autograd.grad(v, p)
    with torch.no_grad():
        v0 = fn(hq.clone())
    print(f"{k:24s} {v.item():12.6f} {v0.item():12.6f} {g.norm().item():12.6f} {str(bool(torch.isfinite(g).all())):>8}")

_, la, lph = L.fourier_space_loss(lq, hq, return_parts=True)
print(f"Fourier Space Loss: амплитудный член {la.item():.4f}, фазовый {lph.item():.4f}")
i, j = L.best_detail_crop(hq, C); print("окно с максимумом деталей:", i, j)
acc = M.band_stats(L.to01(lq), L.to01(hq)); s, m = M.summary(acc, gsd_cm=8.0)
print(M.format_bands(m)); print(s)
