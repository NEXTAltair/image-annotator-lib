# Rating Model Candidates

This document records model-specific facts for rating-capable annotators. The output contract is
defined in ADR 0003; LoRAIro-specific mapping is intentionally out of scope for image-annotator-lib.

## Adopted First: `deepghs/anime_rating`

- Model card: https://huggingface.co/deepghs/anime_rating
- Reference docs: https://dghs-imgutils.deepghs.org/main/api_doc/validate/rating.html
- License: MIT
- Format: ONNX
- Source scheme: `sankaku3`
- Labels: `safe`, `r15`, `r18`
- Default variant: `mobilenetv3_sce_dist`
- Default variant files:
  - `mobilenetv3_sce_dist/model.onnx`
  - `mobilenetv3_sce_dist/meta.json`
- `meta.json` label order: `safe`, `r15`, `r18`
- Model card metrics:
  - `caformer_s36_plus`: 74.26% accuracy, 22.10G FLOPs
  - `mobilenetv3_sce_dist`: 69.49% accuracy, 0.63G FLOPs
- Notes:
  - The model card and imgutils docs state that boundaries between `safe`, `r15`, and `r18`
    are unclear, and the model should be treated as a rough estimate.
  - Use `mobilenetv3_sce_dist` first because it is small enough for routine local validation.
  - LoRAIro maps these labels downstream; the library returns only model-native labels.

## Deferred

| Model | Scheme | Status | Reason |
|---|---|---|---|
| `deepghs/eattach_sankaku_rating` | `sankaku3` | Deferred | Similar labels; compare after `anime_rating` adapter proves the path. |
| `Camais03/camie-tagger-v2` | `danbooru4` | Deferred | Multi-label tagger, likely heavier than rating-only model. |
| `ggg4mless/RateBooru_Efficient` | `danbooru3` | Low priority | TensorFlow/Keras dependency and weaker public metrics. |
| Binary NSFW classifiers | `binary_nsfw` | Deferred | Routing/filter signal rather than fine-grained rating. |
