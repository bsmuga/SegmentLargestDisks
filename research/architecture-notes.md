# Architecture notes — UNet vs. ViT for ranked-disk segmentation

*Last reviewed: 2026-05-03*

## Problem framing

The task in this repo looks like ordinary semantic segmentation — predict a per-pixel class on a binary input image — but its inner structure is different. Per-pixel labels depend on the **size-rank** of the disk that a pixel belongs to among all disks in the image. To label a single pixel, the model must answer three sub-questions:

1. **Is this pixel inside any disk?** Local pattern recognition.
2. **Which disk is it inside, and how big is that disk?** Roughly local — needs a receptive field at least as big as the disk.
3. **Is that disk in the top-K by size?** **Global.** Rank cannot be computed from any local neighborhood; another disk somewhere else in the image, growing or shrinking by one pixel, can flip the answer.

This is the lens through which the architecture choice should be judged. Pixel-classification accuracy hinges on an instance-level, image-global relational comparison.

```
        Per-pixel decision: "what label?"
                     │
        ┌────────────┼─────────────────┐
        ▼            ▼                 ▼
   Inside a      Which disk?       Is that disk
    disk?       (local-ish)        in top-K by size?
   (LOCAL)                          (GLOBAL — needs to
                                   compare against
                                   ALL other disks)
```

### Asymmetric label structure

The data generator (`generate_data.py`, see the loop at `labels[i] = min(i + 1, max_labels)`) produces a non-uniform label scheme. With the defaults in `train.py` (`NUM_CIRCLES=30`, `NUM_LABELED=25`, `MAX_LABELS=5`), labels assigned to disks sorted by descending radius are:

```
rank: 1   2   3   4   5   6   7  ...  25   26   27   28   29   30
label:1   2   3   4   5   5   5  ...   5    0    0    0    0    0
```

So:

- **Labels 1–4** are unique. Each requires the model to identify a *specific* rank — a sharp decision boundary.
- **Label 5** is a wide band covering ranks 5 through 25. Within that band, mistakes are free.
- **Label 0** mixes two distinct populations: true background pixels (the majority of the canvas) and the smallest 5 disks (ranks 26–30).

The asymmetry shapes where errors are likely to land. Labels 1–4 are fragile — the model has to truly rank. Label 5 is forgiving in the middle but fragile at its edges (the rank-4 vs. rank-5 boundary, where a one-pixel-radius difference flips the label, and the rank-25 vs. rank-26 boundary, where the model has to decide "small enough to be in the top 25, but not so small as to fall to background"). Label 0, when it does fall inside a disk, is structurally mixed — a useful signal that "this disk is not labeled" but offers no gradient toward *why*.

## UNet through this lens

```
   Local convs ──▶ pool ──▶ pool ──▶ bottleneck ──▶ unpool ──▶ unpool ──▶ pixel logits
       │                                  │                                    ▲
       └──────────── skip connections ────┴────────────────────────────────────┘
       (carry LOCAL detail)              (sees ~whole image at 128×128 input)
```

UNet's bottleneck on a 128×128 input — with the channel ladder `32 → 64 → 128 → 256 → 512` defined in `models/unet.py` — has a nominal receptive field that spans the whole image. So in principle, a deep enough UNet *can* see all disks at once.

What conv layers do well here:

- **Foreground/background**: trivial. A few conv layers separate "inside any disk" from "outside" with near-perfect accuracy.
- **Single-disk size estimation**: a stack of convs within a disk's footprint can encode "this is part of a disk of radius ~r" in the feature map. Skip connections then carry the position back up to pixel resolution.

Where the inductive bias breaks down:

- **Ranking is not a pattern, it is a comparison.** Conv layers compute *local pattern matches* — they are good at saying "this looks like a disk of radius r" but not at saying "this disk's r is the largest in the image." There is no architectural primitive that compares values across spatial positions.
- The bottleneck features could in principle carry a global summary, but every pixel's prediction has to be reconstructed from skip-connection features (mostly local) plus bottleneck features (heavily compressed). The model has to learn an *implicit sort* through stacked nonlinearities — possible, but brittle.
- **Sharp rank boundaries are the worst case.** A disk that is the 4th-largest by one pixel of radius gets label 4; the 5th-largest gets label 5. The decision boundary in feature space is a step function over a continuous "rank" quantity. CNNs tend to learn smooth thresholds and miscalibrate near the cliff.
- **Skip connections may actively hurt** the global parts of the task. They re-inject high-resolution local features near the output, which biases the head toward decisions made from local context — exactly what is *not* sufficient for ranking.

Predicted profile: **excellent on label 1, very good on labels 2–3, degrading toward 4, and label 5 mostly works because of its wide tolerance band.** Errors will concentrate at the rank-4/5 cliff and along disk edges where pooling-induced blurring meets the integer-radius mask.

## ViT through this lens

```
   16×16 patches ──▶ 64 tokens ──▶ self-attention ──▶ ... ──▶ ConvTranspose ×4 ──▶ logits
                                       │
                                  Every token sees
                                  every other token
                                  from layer 1 — exactly
                                  the right shape for
                                  "compare against all"
```

Self-attention is, almost literally, the operation the problem demands: each patch token computes a weighted summary over all other patch tokens. "Find all the disks, decide which patches are part of the top-K largest" is a natural pattern for attention to learn — the per-pixel decision becomes "based on what every other patch said about its disk, which rank does my patch belong to?"

Where the configuration in `models/vit.py` (`patch_size=16`, `embed_dim=192`, `num_heads=4`, `num_layers=4`, `mlp_ratio=4`) limits what is achievable:

- **Capacity**: ~2.3M parameters total. The encoder must learn (a) recognize disks within a patch, (b) estimate a disk's size, (c) aggregate across patches that share a disk, and (d) rank disks globally — all from pixel input. That is a lot of subroutines for a 4-layer/192-dim model to compose.
- **Patch granularity**: 16×16 patches on 128×128 input → 64 tokens. A small disk may sit entirely inside one patch; a large disk crosses many. The model has to learn an "are we part of the same disk?" routine across tokens, which attention can do but uses a chunk of its representational budget on.
- **Pixel precision is the decoder's job, not attention's.** The 4× ConvTranspose stack upsamples per-patch decisions back to 128×128. Once attention has decided "this patch belongs to the 2nd-largest disk," the decoder paints pixels accordingly. Disk boundaries that fall mid-patch can only be sharpened to the extent that the patch embedding encodes the disk's silhouette — which is a tall order from a 192-dim vector.
- **BatchNorm in the decoder** (`up1_bn` ... `up4_bn`) re-introduces a batch-coupled normalization scheme, which is a small smell in an otherwise transformer-style stack but works at `BATCH_SIZE >= 8`.
- **Sample efficiency / training**: ViTs are typically more data-hungry and pickier about LR schedules than CNNs of similar size. The current training setup (`Adam`, `lr=1e-3`, no warmup, no weight decay, ~5000 samples) is CNN-shaped, not ViT-shaped.

Predicted profile: **better in principle at the rank-boundary cases** because attention is the right tool for cross-disk comparison, **but the under-parameterized model + naive training schedule may yield worse pixel-level mIoU than UNet in practice**. The win from inductive bias may be eaten by capacity and decoder-induced edge softness.

## Side-by-side

| Capability                                  | UNet                       | ViT (current configuration) |
|---------------------------------------------|----------------------------|------------------------------|
| Find disks (foreground / background)        | ★★★★★                      | ★★★★                         |
| Measure single-disk size                    | ★★★★                       | ★★★                          |
| **Compare across all disks (rank)**         | ★★ (struggles)             | ★★★★★ (natural fit)          |
| Pixel-precise boundaries                    | ★★★★★                      | ★★★ (decoder dependent)      |
| Sample efficiency                           | ★★★★                       | ★★                           |
| Training stability at this scale            | ★★★★                       | ★★                           |
| Failure mode at rank-4 / rank-5 cliff       | likely miscalibrated       | likely sharper               |
| Failure mode at disk boundaries             | tight                      | softer (16×16 patch quanta)  |

## Alternatives that bound the deep-learning approach

The task admits a trivial classical solution:

```python
# Pseudocode — not in the repo
labels_img = scipy.ndimage.label(binary_image)[0]
areas = np.bincount(labels_img.ravel())[1:]
order = np.argsort(-areas)
mask = np.zeros_like(labels_img)
for new_label, old_label in enumerate(order, start=1):
    mask[labels_img == old_label] = min(new_label, MAX_LABELS) if new_label <= NUM_LABELED else 0
```

Connected components + sort by area + clamp gives ~100% accuracy by construction (disks are non-overlapping by generation). This means the experiment's *interesting* question is not "can we solve this" — it is **"can a learned model recover the global ranking from pixel labels alone, without being given connected-components as scaffolding?"** That reframes both UNet and ViT results: success is informative about what each architecture's inductive bias can absorb; failure is informative about where the bias falls short.

A hybrid worth considering if the question becomes "what *should* solve this":

```
   ┌──────────────────────────────────────────────────────────┐
   │ UNet encoder  ──▶  Transformer bottleneck  ──▶  UNet decoder │
   │                                                          │
   │ - encoder: extract per-disk features + spatial detail    │
   │ - bottleneck: rank across disks via attention            │
   │ - decoder: paint pixel-precise labels using skip features│
   └──────────────────────────────────────────────────────────┘
```

This pattern (sometimes called "TransUNet" in the literature) directly maps to the three sub-problems: convolutional encoder for local pattern recognition, attention for the global comparison, convolutional decoder + skip connections for pixel precision. Probably overkill for the project's stated minimalism, but it is the cleanest architectural answer to the problem as framed.

## Open questions / what would change this analysis

- **Would more capacity flip ViT's practical result?** Bumping `embed_dim` from 192 to 384 or `num_layers` from 4 to 8 would put the ViT in a regime where the inductive-bias advantage might dominate over capacity limits. Worth a controlled study.
- **Would replacing BatchNorm with LayerNorm/GroupNorm change the comparison?** It would decouple training from `BATCH_SIZE` (currently locked at 8 to keep BatchNorm valid; see `openspec/changes/fix-and-finish-baseline`). LayerNorm is also more natural in the ViT decoder.
- **Does the asymmetric label scheme paper over UNet's ranking weakness?** Label 5's wide tolerance band may be what makes UNet look acceptable on overall mIoU. A stricter scheme (e.g., `MAX_LABELS=10` or `NUM_LABELED=10`) would force the model to rank more granularly and likely widen the gap.
- **Would reframing as instance segmentation + per-instance ranking head change the picture?** Segmentation-then-rank decouples the local and global parts of the task and matches how the classical baseline solves it. It is a different model class, but the closest deep-learning analogue to the structure of the problem.
- **What does training-curve shape say about each architecture?** A ViT that converges on labels 1–3 but stalls on 4–5 (or vice versa) would be diagnostic. The current `train.py` already logs per-class IoU; future runs should keep that visible.
