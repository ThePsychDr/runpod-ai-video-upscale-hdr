# Planned Features

## v1.2.0 — HDR Quality Improvements

- **Optimized recommended settings** from production testing (music videos, 1080p/4K sources)
- **`highlight_boost` documentation** — specular highlight expansion for HDR pop (already implemented, needs docs + recommended defaults)
- **`temporal_smooth` tuning** — reduced from 0.4 to 0.1 for more responsive scene transitions while avoiding brightness pumping

## TensorRT Acceleration (private builds only)

TRT engines are GPU-architecture-specific — an engine built on Ada (4090) won't run on Hopper (H100) or Ampere (A40). Since serverless assigns arbitrary GPU types, TRT acceleration is not viable for public Hub releases. TRT support is available in the [private build](https://github.com/ThePsychDr/AI-upscale-and-HDR) for fixed-GPU deployments (pods, local).

## Future

- **Multi-worker chunk splitting** — automatic job splitting across workers for parallel processing of long videos
- **Adaptive denoise from source analysis** — detect source bitrate and compression artifacts to auto-tune `dn` strength per-scene rather than per-video
- **Audio passthrough** — preserve original audio track through the pipeline (currently strips audio, must be rejoined locally)
