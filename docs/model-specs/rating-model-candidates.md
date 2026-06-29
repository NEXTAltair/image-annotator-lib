---
type: Reference
title: Rating Model Candidates
status: Accepted
timestamp: 2026-05-21
tags: [rating, model-selection]
depends_on: [torch]
---
# Rating Model Candidates

This document records model-specific facts for rating-capable annotators. The output contract is
defined in ADR 0003; LoRAIro-specific mapping is intentionally out of scope for image-annotator-lib.

## Adopted Rating-Only Models: `deepghs/anime_rating`

- Model card: https://huggingface.co/deepghs/anime_rating
- Reference docs: https://dghs-imgutils.deepghs.org/main/api_doc/validate/rating.html
- License: MIT
- Format: ONNX
- Source scheme: `sankaku3`
- Labels: `safe`, `r15`, `r18`
- Fast/default variant: `mobilenetv3_sce_dist`
- Accuracy-first variant: `caformer_s36_plus`
- Variant files:
  - `mobilenetv3_sce_dist/model.onnx`
  - `mobilenetv3_sce_dist/meta.json`
  - `caformer_s36_plus/model.onnx`
  - `caformer_s36_plus/meta.json`
- `meta.json` label order: `safe`, `r15`, `r18`
- Model card metrics:
  - `caformer_s36_plus`: 74.26% accuracy, 22.10G FLOPs
  - `mobilenetv3_sce_dist`: 69.49% accuracy, 0.63G FLOPs
- Notes:
  - The model card and imgutils docs state that boundaries between `safe`, `r15`, and `r18`
    are unclear, and the model should be treated as a rough estimate.
  - Use `mobilenetv3_sce_dist` when speed and batch throughput matter.
  - Use `caformer_s36_plus` when rating accuracy is more important than runtime cost.
  - LoRAIro maps these labels downstream; the library returns only model-native labels.

## Adopted as Large Tagger Candidate: `Camais03/camie-tagger`

- Model card: https://huggingface.co/Camais03/camie-tagger
- ONNX optimized export: https://huggingface.co/deepghs/camie_tagger_onnx
- Reference docs: https://dghs-imgutils.deepghs.org/main/api_doc/tagging/camie.html
- License: GPL-3.0
- Format: ONNX / PyTorch
- Source scheme: `danbooru4`
- Labels: `general`, `sensitive`, `questionable`, `explicit` for rating
- Files in original repo:
  - `model_initial.onnx`
  - `model_initial_metadata.json`
- Library model id: `camie_tagger_initial`
- Objective indicators:
  - Hugging Face likes: about 66 on the original repo when reviewed.
  - Original ONNX size: about 856 MB for `model_initial.onnx`.
  - DeepGHS ONNX export reports 70,527 classes.
  - DeepGHS ONNX export reports F1 micro/macro around 0.584/0.383 for `initial` and
    0.589/0.387 for `refined`.
- Notes:
  - Treat Camie as a large tagger, not as a rating-only model.
  - Initial adapter output should be limited to `rating`, `general`, and `character`.
  - `artist`, `copyright`, `meta`, and `year` are not default outputs because the reference docs
    describe those categories as limited accuracy.
  - Do not copy GPL implementation code or bundle model files. Use the same external ONNX download
    style as WDTagger.

## Deferred

| Model | Scheme | Status | Reason |
|---|---|---|---|
| `deepghs/eattach_sankaku_rating` | `sankaku3` | Rejected for now | Similar labels to `anime_rating`; weaker objective case than `anime_rating` variants. |
| `ggg4mless/RateBooru_Efficient` | `danbooru3` | Low priority | TensorFlow/Keras dependency and weaker public metrics. |
| Binary NSFW classifiers | `binary_nsfw` | Deferred | Routing/filter signal rather than fine-grained rating. |
