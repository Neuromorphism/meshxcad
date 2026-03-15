# Latest Mesh-to-CAD Research and Open-Weight Targets

This repo now includes local subprojects for three practical mesh-to-CAD methods:

1. `CAD-Recode v1.5`
2. `cadrille`
3. `Point2CAD`

## Why these three

- `CAD-Recode` is a point-cloud-to-CadQuery model with open Hugging Face checkpoints.
- `cadrille` is the strongest current open-weight point-cloud CAD-code model with official inference code.
- `Point2CAD` remains a useful open model for surface/topology recovery from segmented point clouds and runs locally through the published Docker image.

## Current research snapshot

### 1. DreamCAD

- Paper date: March 5, 2026
- Project: https://sadilkhan.github.io/dreamcad2026/
- Paper: https://arxiv.org/abs/2603.05607
- Notes:
  - Newest point/image/text-to-CAD paper I found.
  - Produces editable parametric surfaces and STEP output.
  - The project page says code and dataset will be released publicly.
  - I did not integrate it here because I could not confirm released inference code or open model weights as of March 13, 2026.

### 2. cadrille

- Paper: https://arxiv.org/abs/2505.22914
- Repo: https://github.com/col14m/cadrille
- Models:
  - https://huggingface.co/maksimko123/cadrille
  - https://huggingface.co/maksimko123/cadrille-rl
- Why integrated:
  - Open repository.
  - Open SFT and RL checkpoints.
  - Official point-cloud inference path.

### 3. CAD-Recode

- Paper: https://arxiv.org/abs/2412.14042
- Repo: https://github.com/filaPro/cad-recode
- Models:
  - https://huggingface.co/filapro/cad-recode
  - https://huggingface.co/filapro/cad-recode-v1.5
- Why integrated:
  - Open checkpoints.
  - Small enough to fit comfortably on an RTX 5090.
  - Directly emits CadQuery Python that can be exported to STEP and STL.

### 4. Point2CAD

- Repo: https://github.com/prs-eth/point2cad
- Project page: https://www.obukhov.ai/point2cad
- Docker image: https://hub.docker.com/r/toshas/point2cad
- Why integrated:
  - Still the cleanest open local pipeline for segmented point-cloud to CAD-style primitive surface reconstruction.
  - Has official Docker packaging and bundled ParseNet checkpoints.
  - Best treated as a geometry/topology reconstruction stage, not a full CAD program generator.

## Why not DeepCAD here

`DeepCAD` is still an important baseline, but the repository is centered on CAD-sequence autoencoding and latent generation. Its `pc2cad` path exists, but the public docs are weaker on ready-to-run point-cloud reconstruction checkpoints than the three targets above.

## GPU fit

These integrations were chosen to be realistic on an RTX 5090 class GPU:

- `CAD-Recode v1.5`: Qwen2-1.5B backbone plus point encoder.
- `cadrille`: Qwen2-VL-2B-based multimodal model, point-cloud mode supported.
- `Point2CAD`: lightweight fitting pipeline, usually much smaller than the LLM-based models.
