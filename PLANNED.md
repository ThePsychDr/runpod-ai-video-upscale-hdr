# Planned Features

## v1.2.0 — HDR Quality Improvements

- **Optimized recommended settings** from production testing (music videos, 1080p/4K sources)
- **`highlight_boost` documentation** — specular highlight expansion for HDR pop (already implemented, needs docs + recommended defaults)
- **`temporal_smooth` tuning** — reduced from 0.4 to 0.1 for more responsive scene transitions while avoiding brightness pumping

## TensorRT Acceleration (private builds only)

TRT engines are GPU-architecture-specific — an engine built on Ada (4090) won't run on Hopper (H100) or Ampere (A40). Since serverless assigns arbitrary GPU types, TRT acceleration is not viable for public Hub releases. TRT support is being testeded on fixed-GPU deployments (pods, local).

## Implemented

- **Manual chunk splitting** — `start_time` and `chunk_duration` parameters allow processing a segment of the video. Users can manually split a long video into chunks, submit each as a separate job, and concatenate outputs locally with `ffmpeg -f concat`
- **Per-video adaptive denoise** — `dn: -1` auto-detects optimal denoise strength from source resolution + bitrate via `recommend_dn()`. Applied uniformly to the entire video

## Planned

### Auto Job Orchestration

Handler receives a single job, calculates chunks based on video duration and available workers, submits N sub-jobs via the RunPod API, polls for completion, and concatenates the output segments into a single file. Eliminates the need for manual chunk coordination.

### Per-Scene Adaptive Denoise (research)

Vary DN strength within a single video based on per-frame compression artifact analysis. Current `dn: -1` applies one value to the entire video — scenes with different compression quality (e.g. dark scenes with more blocking vs bright scenes with less) would benefit from per-frame adaptation.

**Approach under investigation:**
- Analyze each frame's compression artifact density before Stage 1 (variance of Laplacian for blur/blocking, DCT coefficient analysis for quantization artifacts)
- Map artifact severity to DN strength (heavy blocking → DN 0.7, clean frame → DN 0.15)
- Smooth DN values with EMA (like `TemporalSmoother`) to avoid jarring quality shifts at scene cuts
- Feasible because `run_stage1()` accepts DN per-call — it controls the blend weight between the general model and the WDN (weighted denoise) model, no model reload needed
