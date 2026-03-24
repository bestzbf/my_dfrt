#!/usr/bin/env python3
"""GPU performance profiler for D4RT training pipeline."""
import sys
import time
import torch
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models import create_d4rt
from data import PointOdysseyDataset, collate_fn
from torch.utils.data import DataLoader

# B300 (sm_103) workaround: disable flash/mem_efficient SDP on Blackwell
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    if cap[0] >= 10:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print(f"B300 detected (sm_{cap[0]}{cap[1]}): using math SDP backend")

DEVICE = torch.device("cuda:0")
DATA_ROOT = os.environ.get("DATA_ROOT", "/data2/d4rt/datasets/PointOdyssey_fast")
VIDEOMAE_MODEL = os.environ.get("VIDEOMAE_MODEL", "/data1/zbf/pretrained/videomae-base")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "32"))

def timer(label, fn, warmup=2, runs=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / runs * 1000
    mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
    print(f"  {label:<40} {elapsed:8.1f} ms   peak_mem={mem:.2f}GB")
    torch.cuda.reset_peak_memory_stats(DEVICE)
    return elapsed

def main():
    print(f"\n{'='*60}")
    print(f"D4RT GPU Profiler  |  device={DEVICE}  bs={BATCH_SIZE}")
    print(f"{'='*60}\n")

    # --- Build model ---
    print("Building model...")
    model = create_d4rt(
        encoder_variant="base",
        img_size=256,
        temporal_size=48,
        decoder_depth=6,
        videomae_model=VIDEOMAE_MODEL,
        patch_provider="precomputed_highres",
    ).to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M\n")

    # --- Fake batch ---
    B = BATCH_SIZE
    T = 48
    H = W = 256
    N = 2048
    patch_size = 9

    video   = torch.randn(B, T, 3, H, W, device=DEVICE)
    coords  = torch.rand(B, N, 2, device=DEVICE)
    t_src   = torch.randint(0, T, (B, N), device=DEVICE)
    t_tgt   = torch.randint(0, T, (B, N), device=DEVICE)
    t_cam   = torch.randint(0, T, (B, N), device=DEVICE)
    aspect  = torch.ones(B, 1, device=DEVICE)
    patches = torch.randn(B, N, 3, patch_size, patch_size, device=DEVICE)

    print("=" * 60)
    print("  Stage                                    Time(ms)   VRAM")
    print("=" * 60)

    # 1. Encoder only
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        enc_feat = model.encode(video, aspect)  # warmup
    def run_encoder():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model.encode(video, aspect)
    t_enc = timer("Encoder (encode)", run_encoder)

    # 2. Decoder only (reuse enc_feat)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        enc_feat = model.encode(video, aspect)
    def run_decoder():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model.decode(
                encoder_features=enc_feat,
                frames=video,
                coords=coords,
                t_src=t_src,
                t_tgt=t_tgt,
                t_cam=t_cam,
                local_patches=patches,
            )
    t_dec = timer("Decoder (decode, N=2048)", run_decoder)

    # 3. Full forward
    def run_forward():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feat = model.encode(video, aspect)
            return model.decode(
                encoder_features=feat,
                frames=video,
                coords=coords,
                t_src=t_src,
                t_tgt=t_tgt,
                t_cam=t_cam,
                local_patches=patches,
            )
    t_fwd = timer("Full forward (enc+dec)", run_forward)

    # 4. Forward + backward (training mode)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    def run_fwd_bwd():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feat = model.encode(video, aspect)
            out = model.decode(
                encoder_features=feat,
                frames=video,
                coords=coords,
                t_src=t_src,
                t_tgt=t_tgt,
                t_cam=t_cam,
                local_patches=patches,
            )
            loss = out["pos_3d"].mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    t_bwd = timer("Forward + Backward (AMP)", run_fwd_bwd, warmup=1, runs=3)

    # 5. Without AMP
    def run_fwd_bwd_fp32():
        optimizer.zero_grad(set_to_none=True)
        feat = model.encode(video, aspect)
        out = model.decode(
            encoder_features=feat,
            frames=video,
            coords=coords,
            t_src=t_src,
            t_tgt=t_tgt,
            t_cam=t_cam,
            local_patches=patches,
        )
        loss = out["pos_3d"].mean()
        loss.backward()
        optimizer.step()

    t_fp32 = timer("Forward + Backward (FP32)", run_fwd_bwd_fp32, warmup=1, runs=3)

    # 6. GPU utilization estimate
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Encoder time:        {t_enc:8.1f} ms  ({t_enc/t_fwd*100:.0f}% of forward)")
    print(f"  Decoder time:        {t_dec:8.1f} ms  ({t_dec/t_fwd*100:.0f}% of forward)")
    print(f"  AMP fwd+bwd:         {t_bwd:8.1f} ms")
    print(f"  FP32 fwd+bwd:        {t_fp32:8.1f} ms")
    print(f"  AMP speedup:         {t_fp32/t_bwd:.2f}x")
    print(f"  Throughput (AMP):    {B*1000/t_bwd:.1f} samples/sec")
    print(f"  Throughput (FP32):   {B*1000/t_fp32:.1f} samples/sec")

    # 7. Decoder scaling with N
    print(f"\n  Decoder scaling with num_queries:")
    model.eval()
    with torch.no_grad():
        enc_feat = model.encode(video, aspect)
    for n in [256, 512, 1024, 2048, 4096]:
        c = torch.rand(B, n, 2, device=DEVICE)
        ts = torch.randint(0, T, (B, n), device=DEVICE)
        p = torch.randn(B, n, 3, patch_size, patch_size, device=DEVICE)
        def _run(n=n, c=c, ts=ts, p=p):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return model.decode(enc_feat, video, c, ts, ts, ts, p)
        t = timer(f"  Decoder N={n}", _run, warmup=1, runs=3)

    print(f"\n{'='*60}")
    print("Recommendations:")
    if t_bwd < t_fp32 * 0.8:
        print("  ✅ AMP gives significant speedup — keep --amp")
    else:
        print("  ⚠️  AMP speedup is small — check BF16 support")
    if t_enc > t_dec:
        print("  ⚠️  Encoder is the bottleneck — consider gradient checkpointing OFF")
    else:
        print("  ⚠️  Decoder is the bottleneck — consider smaller num_queries or query chunking")
    print(f"  💡 Data loading (401 samp/s) vs GPU ({B*1000/t_bwd:.0f} samp/s)")
    if B*1000/t_bwd > 401:
        print("  ⚠️  GPU faster than data loading — increase NUM_WORKERS further")
    else:
        print("  ✅ Data loading is not the bottleneck")
    print()

if __name__ == "__main__":
    main()
