# Planned Features

## Implemented

### v1.7.x — Consolidation + GPU Compatibility

- **Repo consolidation** — 4 repos merged into 2 (this public repo + private local/TRT repo)
- **`ADA_24` pool ID** for Hub tests (replaces legacy `NVIDIA GeForce RTX 4090` name)
- **Removed CACHEBUST** — was causing infinite Hub rebuilds
- **GPU compatibility docs** — full pool mapping, Blackwell explanation, NVENC/NVDEC details
- **`no_denoise` flag** — skip first-stage artifact cleanup

### v1.2.0 — HDR Quality + Stability

- **Tiled ITM inference** — 4K HDR on 24GB GPUs without OOM (~15x faster than guide fallback)
- **`highlight_boost`** — specular highlight expansion for HDR pop (0.0–1.0)
- **`temporal_smooth`** — reduce frame-to-frame flicker (default 0.4)
- **CUDA runtime API Docker build** — no nvcc required, smaller image

### v1.1.0 — Core Features

- **Manual chunk splitting** — `start_time` + `chunk_duration` for multi-GPU parallel processing
- **Per-video adaptive denoise** — `dn: -1` auto-detects from resolution + bitrate
- **Film grain preservation** — `preserve_grain` + `grain_strength`

## TensorRT Acceleration (private builds only)

TRT engines are GPU-architecture-specific — an engine built on Ada (4090) won't run on Hopper (H100) or Ampere (A40). Since serverless assigns arbitrary GPU types from a pool, TRT acceleration is not viable for public Hub releases.

Pre-built engines per architecture (sm_86, sm_89, sm_90) COPYed into the Docker image would solve this but is not yet implemented. TRT support is tested on fixed-GPU deployments (pods, local).

## Planned

### Per-Scene Adaptive Denoise (research)

Vary DN strength within a single video based on per-frame compression artifact analysis. Current `dn: -1` applies one value to the entire video — scenes with different compression quality (e.g. dark scenes with more blocking vs bright scenes with less) would benefit from per-frame adaptation.

**Approach under investigation:**
- Analyze each frame's compression artifact density before Stage 1 (variance of Laplacian for blur/blocking, DCT coefficient analysis for quantization artifacts)
- Map artifact severity to DN strength (heavy blocking → DN 0.7, clean frame → DN 0.15)
- Smooth DN values with EMA (like `TemporalSmoother`) to avoid jarring quality shifts at scene cuts
- Feasible because `run_stage1()` accepts DN per-call — it controls the blend weight between the general model and the WDN (weighted denoise) model, no model reload needed

### Blackwell Support (waiting on RunPod)

RTX 5090/5080 and B200 require PyTorch 2.11+ for sm_100/sm_120 CUDA kernels. Blocked until RunPod releases an official serverless base image with PyTorch 2.11+. Current latest official image is PyTorch 2.8. No code changes needed — just a base image swap.
