# KL-iT LLM Sizing Calculator

Free, static (no backend) calculator that estimates GPU VRAM and tokens/sec for LLM **inference** and **fine-tuning** (LoRA / QLoRA / full) across 55+ workstation, GeForce, datacenter and unified-memory GPUs. Thai / English.

Live: https://llmsizingtool.kl-itonline.com  ·  By [KL-iT ONLINE](https://www.kl-itonline.com) (KLMO Group Co., Ltd.)

## Files
- `index.html` — the whole app (HTML + CSS + JS, single file). GPU/model catalogs are the `GPUS` and `MODELS` arrays near the top of the script.
- `og-image.png` — social share image.
- `CNAME` — custom domain for GitHub Pages.

## URL presets
`?gpu=pro6000` · `?model=l3.3-70b` · `?mode=ft` · `?lang=en` — ids are the `id` fields in the catalogs.

## Method
Public formulas only: weights = params × bytes/param; KV = 2 × layers × KV-heads × head-dim × tokens (MLA and sliding-window handled); fine-tune = weights + gradients + optimizer states + activations (ZeRO-3/FSDP sharding across GPUs); decode speed ≈ memory-bandwidth bound; training ≈ 6 × params FLOPs/token at 25–40% MFU. Estimates only — expect ±10% on VRAM, 15–25% on speed.
