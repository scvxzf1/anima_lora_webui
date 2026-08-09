"""R-verify: NF4 权重磁盘落盘/加载 round-trip + 真实 DiT 落地.

验证 quantize.py 的 save_nf4_dit / load_nf4_dit_into 用 bnb 0.49.2 官方
QuantState.as_dict(packed=True) + from_prequantized 契约 round-trip 保真.

阶段 1 (小规模冒烟): 2 个 Linear4bit 量化 -> save -> load -> forward delta=0,
  验证 bnb safetensors 序列化契约 (含 state2 双重量化).
阶段 2 (真实 DiT 落地): load_krea2_dit(nf4=True) 在线量化 (PG199) ->
  save_nf4_dit 落地 models/diffusion_models/krea2_raw_nf4.safetensors (~6.6GB) ->
  load_nf4_dit_into 从磁盘加载 -> forward 对比在线量化版, delta 应很小 (同量化
  数据, 仅路径不同). 落地后 3080 等小卡可直接 load_nf4_dit_into 绕过在线量化.

环境变量:
  K2_NF4SAVE_GPU=0/1   (默认 1=PG199, 阶段2 在线量化需大卡)
  K2_NF4SAVE_PHASE=1/2/all (默认 all: 1 冒烟 + 2 真实落地)
  K2_NF4SAVE_OUT=path  (默认 models/diffusion_models/krea2_raw_nf4.safetensors)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2_NF4SAVE_GPU", "1"))

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bitsandbytes.nn import Linear4bit, Params4bit  # noqa: E402
from library.models.krea2_raw.quantize import (  # noqa: E402
    load_nf4_dit_into,
    save_nf4_dit,
)

QUANT_TYPE = "nf4"
COMPRESS_STATISTICS = True
COMPUTE_DTYPE = torch.bfloat16
DEFAULT_OUT = ROOT / "models" / "diffusion_models" / "krea2_raw_nf4.safetensors"


def _make_linear4bit(out_f: int, in_f: int, device, dtype) -> Linear4bit:
    g = torch.Generator(device="cpu").manual_seed(42)
    lin = Linear4bit(
        in_f, out_f, bias=True, compute_dtype=dtype,
        quant_type=QUANT_TYPE, compress_statistics=COMPRESS_STATISTICS,
    )
    with torch.no_grad():
        lin.weight = Params4bit(
            torch.randn(out_f, in_f, generator=g, dtype=dtype),
            requires_grad=False, quant_type=QUANT_TYPE,
            compress_statistics=COMPRESS_STATISTICS, module=lin,
        )
        lin.bias.data = torch.randn(out_f, generator=g, dtype=dtype) * 0.01
    lin = lin.to(device)
    assert lin.weight.bnb_quantized, "量化未触发"
    return lin


class _TinyNF4Model(nn.Module):
    """2 个 Linear4bit 的小模型, 验证 save/load round-trip."""

    def __init__(self, device, dtype) -> None:
        super().__init__()
        self.fc1 = _make_linear4bit(512, 1024, device, dtype)
        self.fc2 = _make_linear4bit(256, 512, device, dtype)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


def phase1_smoke(tmp_path) -> int:
    print("\n=== 阶段 1: 小规模 save/load round-trip 冒烟 ===")
    device = torch.device("cuda")
    dtype = COMPUTE_DTYPE

    orig = _TinyNF4Model(device, dtype).to(dtype)
    # 参考输入 + 参考 forward
    g = torch.Generator(device=device).manual_seed(7)
    x = torch.randn(4, 1024, generator=g, device=device, dtype=dtype)
    with torch.no_grad():
        ref_out = orig(x).clone()
    ref_fc1_data = orig.fc1.weight.data.clone()
    ref_fc2_data = orig.fc2.weight.data.clone()

    # 存盘
    out_path = str(tmp_path / "tiny_nf4.safetensors")
    info = save_nf4_dit(orig, out_path)
    print(f"  存盘: {info['linear4bit_count']} 个 Linear4bit, "
          f"{info['bytes'] / 1e6:.1f}MB -> {out_path}")

    # 加载: 构造同结构空 bf16 模型, 用 load_nf4_dit_into 填回
    loaded = _TinyNF4Model(device, dtype).to(dtype)
    # 先把 fc1/fc2 重置成普通 nn.Linear 空壳 (load_nf4_dit_into 会替换成 Linear4bit)
    loaded.fc1 = nn.Linear(1024, 512, bias=True)
    loaded.fc2 = nn.Linear(512, 256, bias=True)
    loaded = loaded.to(dtype)
    n = load_nf4_dit_into(loaded, out_path, torch.device("cpu"))
    print(f"  加载: {n} 个 Linear4bit")
    loaded = loaded.to(device).to(dtype)

    # 验证
    assert isinstance(loaded.fc1, Linear4bit), "fc1 未替换成 Linear4bit"
    assert loaded.fc1.weight.bnb_quantized, "fc1 bnb_quantized 丢失"
    # 4-bit 码逐字节一致 (同量化数据, 仅路径不同)
    data_match_fc1 = torch.equal(loaded.fc1.weight.data.cpu(), ref_fc1_data.cpu())
    data_match_fc2 = torch.equal(loaded.fc2.weight.data.cpu(), ref_fc2_data.cpu())
    with torch.no_grad():
        loaded_out = loaded(x)
    delta = (loaded_out - ref_out).abs().max().item()
    print(f"  fc1 4-bit 码一致: {data_match_fc1}")
    print(f"  fc2 4-bit 码一致: {data_match_fc2}")
    print(f"  forward max delta: {delta:.2e} (容差 1e-6, 同数据应 0)")

    ok = data_match_fc1 and data_match_fc2 and delta < 1e-5
    print(f"  阶段 1 通过: {ok}")
    return 0 if ok else 1


def phase2_real_dit(out_path: Path) -> int:
    print("\n=== 阶段 2: 真实 Krea-2 DiT 在线量化 -> 落盘 -> 磁盘加载对比 ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    from library.models.krea2_raw.weights import load_krea2_dit  # noqa: E402
    from library.models.krea2_raw.family import (  # noqa: E402
        Krea2TextEmbedding,
        forward_for_loss,
    )
    from library.models.krea2_raw.strategy import (  # noqa: E402
        Krea2TextEncodingStrategy,
        Krea2TokenizeStrategy,
        load_krea2_text_encoder,
    )
    from library.models.qwen_vae import load_vae  # noqa: E402

    KREA2_DIT = ROOT / "models" / "diffusion_models" / "krea2_raw_bf16.safetensors"
    KREA2_TE = ROOT / "models" / "text_encoders" / "qwen3vl_4b_bf16.safetensors"
    KREA2_VAE = ROOT / "models" / "vae" / "qwen_image_vae.safetensors"
    PROMPT = "a red circle on blue background"
    IMG_SIZE = 512  # 对比 forward 用, 不跑训练

    # === A. 在线量化版 (PG199) ===
    print("--- A. 在线量化 DiT (load_krea2_dit nf4=True) ---")
    t0 = time.time()
    dit_online = load_krea2_dit(KREA2_DIT, device=device, dtype=dtype, eval=True, nf4=True)
    print(f"  在线量化: {time.time()-t0:.1f}s")

    # 落盘
    print(f"--- B. 落盘到 {out_path} ---")
    t1 = time.time()
    save_info = save_nf4_dit(dit_online, str(out_path))
    print(f"  落盘: {save_info['linear4bit_count']} Linear4bit, "
          f"{save_info['bytes'] / 1e9:.2f}GB, {time.time()-t1:.1f}s")

    # 取在线版一个 Linear4bit 的 4-bit 码 + forward 参考
    l4_online = next(m for _, m in dit_online.named_modules() if isinstance(m, Linear4bit))
    ref_data = l4_online.weight.data.clone()

    # === C. 磁盘加载版 (空 bf16 结构 + load_nf4_dit_into, 不重新量化) ===
    print("--- C. 磁盘加载 (load_krea2_dit nf4=False + load_nf4_dit_into) ---")
    t2 = time.time()
    dit_disk = load_krea2_dit(KREA2_DIT, device="cpu", dtype=dtype, eval=True, nf4=False)
    n_loaded = load_nf4_dit_into(dit_disk, str(out_path), torch.device("cpu"))
    print(f"  磁盘加载: {n_loaded} Linear4bit, {time.time()-t2:.1f}s (未重新量化)")
    dit_disk = dit_disk.to(device).to(dtype)

    # 验证: 取同 path 的 Linear4bit, 4-bit 码逐字节一致
    l4_disk = None
    for name, m in dit_disk.named_modules():
        if isinstance(m, Linear4bit):
            l4_disk = m
            disk_path = name
            break
    data_match = torch.equal(l4_disk.weight.data.cpu(), ref_data.cpu())
    print(f"  同 path 4-bit 码一致: {data_match}")
    print(f"  bnb_quantized (disk): {l4_disk.weight.bnb_quantized}")

    # === D. forward 数值对比 (用 VAE latent + TE hiddens) ===
    print("--- D. forward 数值对比 (在线量化 vs 磁盘加载) ---")
    te_model, tokenizer = load_krea2_text_encoder(str(KREA2_TE), dtype=dtype, device="cuda")
    tok = Krea2TokenizeStrategy()
    enc = Krea2TextEncodingStrategy()
    with torch.inference_mode():
        [hiddens, txtmask] = enc.encode_tokens(tok, [te_model], tok.tokenize([PROMPT]))
    del te_model, tok, enc
    torch.cuda.empty_cache()
    vae = load_vae(str(KREA2_VAE), device=device, dtype=dtype, eval=True)
    ys = torch.linspace(0, 1, IMG_SIZE, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, IMG_SIZE, device=device, dtype=dtype)
    pixels = torch.stack([ys.view(-1, 1).expand(IMG_SIZE, IMG_SIZE),
                          xs.view(1, -1).expand(IMG_SIZE, IMG_SIZE),
                          ((xs.view(1, -1).expand(IMG_SIZE, IMG_SIZE) * 8).int().float() % 2)], dim=0).unsqueeze(0)
    pixels = (pixels * 0.8 + 0.1).clamp(0, 1)
    with torch.no_grad():
        latents_4d = vae.encode_pixels_to_latents(pixels)
    del vae, pixels
    torch.cuda.empty_cache()
    latents_5d = latents_4d.unsqueeze(2).to(device)
    text_emb = Krea2TextEmbedding(hiddens.to(device), txtmask.to(device))
    sigma = torch.full((1,), 0.5, device=device, dtype=dtype)
    with torch.no_grad():
        vel_online = forward_for_loss(dit_online, latents_5d, text_emb, sigma).clone()
        vel_disk = forward_for_loss(dit_disk, latents_5d, text_emb, sigma)
    delta = (vel_disk - vel_online).abs().max().item()
    rel = (vel_disk - vel_online).norm().item() / vel_online.norm().item()
    finite = torch.isfinite(vel_disk).all().item()
    print(f"  velocity online shape: {tuple(vel_online.shape)}")
    print(f"  forward max delta: {delta:.2e}")
    print(f"  forward rel L2: {rel:.2e} (同量化数据, 应 ~0)")

    ok = data_match and finite and delta < 1e-3
    print(f"\n  落盘文件: {out_path} ({save_info['bytes'] / 1e9:.2f}GB)")
    print(f"  阶段 2 通过: {ok}")
    return 0 if ok else 1


def main() -> int:
    phase = os.environ.get("K2_NF4SAVE_PHASE", "all")
    out_path = Path(os.environ.get("K2_NF4SAVE_OUT", str(DEFAULT_OUT)))
    print(f"=== NF4 权重落盘/加载验证 (phase={phase}, GPU={os.environ.get('CUDA_VISIBLE_DEVICES')}) ===")

    rc = 0
    if phase in ("1", "all"):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            rc1 = phase1_smoke(Path(tmp))
            rc = rc or rc1
    if phase in ("2", "all"):
        rc2 = phase2_real_dit(out_path)
        rc = rc or rc2
    print(f"\n=== 总结果: {'通过' if rc == 0 else '失败'} ===")
    return rc


if __name__ == "__main__":
    sys.exit(main())
