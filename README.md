# Intracranial Hemorrhage Detection and Explainability

## Overview
Forensic pathology departments deal with heavy caseloads and not enough specialists, and post-mortem CT (PMCT) is expensive to annotate. Clinical CT isn't. The RSNA ICH dataset gives us hundreds of thousands of labeled slices, so training on clinical data first and worrying about PMCT later is the obvious move. But a model that classifies well isn't the same as a model you can trust to explain itself. This repo trains three ICH classifiers on RSNA and checks whether their explanations actually hold up, without needing pixel-level annotations we don't have.

## Research Question
Can a clinically trained ICH model give explanations that are faithful, stable, and consistent enough across architectures to actually help a forensic reviewer, without expensive pixel-level ground truth?

The comparative version of this: across ResNet-50, ViT-B/16, and ConvNeXt-Base, which one gives the strongest evidence on classification, explanation faithfulness (AOPC), explanation stability (Max-Sensitivity), calibration, and the bone-window check? We didn't define an aggregation method going in, so there's no single combined score. We compare metric by metric instead.

## Dataset
- **RSNA ICH CT dataset** — five hemorrhage subtypes (epidural, intraparenchymal, intraventricular, subarachnoid, subdural) plus the aggregate "any" label.
- **Split** — 70/15/15, stratified on "any", `random_state=42`. Train: 188,402. Val: 40,372. Test: 40,373. Val was used for early stopping, per-class thresholds, and calibration. Test was untouched until final metrics.
- **Input representation** — each slice is a 3-channel image built from three CT windows, not a cosmetic colormap: brain (W80/L40), subdural (W200/L80), bone (W2000/L600). ImageNet normalization throughout. Random horizontal flip during training only.
- **Sampling** — `BalancedRandomSampler` equalizes any=1 vs any=0 per epoch. It doesn't touch the subtype balance within any=1, so epidural stays stuck at ~1.18% test prevalence. This matters a lot later.

Since bone is literally one of the three input channels, Grad-CAM could end up tracking skull intensity instead of the actual hemorrhage. That's the whole reason the bone-correlation check exists below.

## Methodology
### Models
All three start from ImageNet pretraining, mostly frozen, fine-tuned with a new `Dropout(0.3) → Linear` head:

| | ResNet-50 | ViT-B/16 | ConvNeXt-Base |
|---|---|---|---|
| Unfrozen depth | layer4 | last encoder block only | last stage only |
| Optimizer | SGD (momentum 0.9) | AdamW | AdamW |
| LR | 1e-3 | 1e-4 | 1e-4 |
| Warmup | none | 1 epoch linear | 1 epoch linear |
| Scheduler | cosine annealing | cosine annealing after warmup | cosine annealing after warmup |
| Best checkpoint | epoch 14 (val loss 0.1905) | epoch 5 (val loss 0.1726) | epoch 13 (val loss 0.1759) |

Worth being upfront: this isn't a clean ablation. Optimizer, LR, warmup, and unfrozen depth all differ, so we can't cleanly blame architecture for every gap we see.

### Evaluation
- **ROC-AUC / PR-AUC** — PR-AUC matters more here because epidural is rare enough (1.18%) that ROC-AUC alone would look artificially good.
- **F1** — at per-class, validation-derived thresholds.
- **Calibration** — temperature scaling fit on val, checked on test, via ECE, MCE, Brier, NLL, both aggregate and per subtype (aggregate improving doesn't mean every subtype improved).
- **Grad-CAM** — via `pytorch_grad_cam`, plain Grad-CAM (not HiResCAM) so it's consistent across CNN and transformer backbones. Target layers: `layer4[-1]` for ResNet-50, `features[-1][-1]` for ConvNeXt-Base, `encoder.layers[-1].ln_1` for ViT-B/16 (CLS token dropped, remaining 196 tokens reshaped to a 14×14 grid).
- **AOPC** — cumulatively perturb the top Grad-CAM patches (16×16 grid, mean-fill, 10 steps) and see how much the predicted probability drops. Tells you if the model actually depends on the region Grad-CAM points to, not whether that region is anatomically correct.
- **Max-Sensitivity** — add small Gaussian noise to the input (r = 0.05σ, 10 draws), recompute Grad-CAM each time, take the max change. Lower = more stable explanation.
- **Bone-window correlation** — Pearson correlation between the Grad-CAM map and the bone channel. Flags possible bone shortcuts, doesn't prove them.

## Results
Macro classification on the held-out test set:

| Model | Macro ROC-AUC | Macro PR-AUC |
|---|---|---|
| ResNet-50 | 0.9186 | 0.6284 |
| ViT-B/16 | 0.9423 | 0.7180 |
| ConvNeXt-Base | 0.9403 | 0.6828 |

Epidural is where things actually separate, since it's the rarest class:

| Metric | ResNet-50 | ViT-B/16 | ConvNeXt-Base |
|---|---|---|---|
| ROC-AUC | 0.8703 | 0.9361 | 0.9291 |
| PR-AUC | 0.0944 | 0.3970 | 0.2513 |
| F1 | 0.1764 | 0.4144 | 0.3136 |
| AOPC | 0.114 ± 0.059 | 0.505 ± 0.206 | 0.338 ± 0.138 |
| Max-Sensitivity | 32.60 | 29.94 | 18.22 |

No clean winner. ViT-B/16 takes aggregate classification, epidural AOPC, and the biggest calibration gain. ConvNeXt-Base takes intraparenchymal/intraventricular AOPC and the lowest Max-Sensitivity across the board. ResNet-50 just isn't competitive on epidural, classification or AOPC, and doesn't make up for it anywhere else.

## Explainability & Reliability Analysis
Good classification numbers don't tell you if the explanation is worth anything, so we ran three separate checks. AOPC tells you if the model actually relies on the region Grad-CAM highlights, but that's dependence, not correctness, a model can depend on the wrong region for a real reason. Max-Sensitivity checks if the map stays put when you add a bit of noise; an explanation that jumps around isn't one a reviewer can lean on. The bone check exists purely because bone is baked into the input, so a strong correlation there is a red flag worth looking at, even though a weak one doesn't clear the model either.

These three don't agree cleanly, which is honestly the interesting part. ViT-B/16 wins epidural AOPC, but a bunch of its Grad-CAM maps are diffuse or drift toward the skull edge, and its bone-correlation numbers are all over the place (17/30 epidural cases go past |r| = 0.2). ConvNeXt-Base's maps sit tighter around plausible hemorrhage regions and its bone correlation is more consistent. ResNet-50 is somewhere in the middle on bone correlation but is just behind everywhere else for epidural. So it's not "ConvNeXt beats ViT," it's that perturbation sensitivity, what the map actually looks like, and bone correlation don't always point the same way, so none of them alone is enough to trust.

**Status:** population-level stats for AOPC and Max-Sensitivity (confidence intervals, proper paired comparisons instead of just comparing means) are still running. Everything above is the point-estimate results from the manuscript.

## Limitations
- Not a controlled ablation, optimizer/LR/warmup/unfrozen-depth all differ, so architecture isn't isolated as the only variable.
- No ground-truth localization masks for RSNA, so we never directly checked if Grad-CAM is anatomically correct, only inferred it indirectly.
- Grad-CAM isn't anatomically validated. AOPC and Max-Sensitivity tell you about model dependence and map stability, not whether the highlighted region is the actual hemorrhage.
- Bone correlation is a spatial-association sanity check, not causal proof either way.
- No statistical inference yet, everything's a point estimate from n = 50 (AOPC/Max-Sens) or n = 30 (bone correlation) per class, and the SDs are big enough that close means shouldn't be read as a settled ranking.
- Slice-level independence across patients is unresolved, so the real amount of independent evidence might be smaller than the slice counts suggest.

## Future Work
- Domain-shift evaluation, clinical → forensic/PMCT transfer. Carrying both ViT-B/16 and ConvNeXt-Base forward instead of picking one now.
- PMCT brings artifacts clinical CT doesn't have (hypostasis, loss of grey-white differentiation, gas, cerebral collapse, later liquefaction). The open question isn't whether classification transfers, prior work already showed that works. It's whether the explanations survive.
- Actual statistical rigor at that stage: confidence intervals, bootstrap intervals for AOPC/Max-Sensitivity, real backbone comparisons instead of eyeballing point estimates, case-level independence handling.

## Repository Structure
```
├── data/          # preprocessed dataset + README describing preprocessing
├── notebooks/      # training and evaluation notebooks per backbone
├── results/        # metrics, calibration, explainability outputs (AOPC, Max-Sensitivity, bone-correlation, Grad-CAM figures) per model
├── LICENSE
└── README.md
```

## Reproducibility
- **Environment**: PyTorch, `pytorch_grad_cam`, standard ImageNet-pretrained weights (`ResNet50_Weights.IMAGENET1K_V2`, `ViT_B_16_Weights.IMAGENET1K_V1`, ConvNeXt-Base ImageNet1K weights).
- **Training**: per-backbone scripts in `notebooks/`. All three use `BCEWithLogitsLoss`, batch size 32, max 15 epochs, patience 5, split fixed at `random_state=42`.
- **Evaluation**: F1-optimal thresholds picked on val, applied to test. Calibration fit on val, checked on test.
- **Explainability**: Grad-CAM, AOPC, Max-Sensitivity computed on confirmed true positives sampled without replacement (`random_state=42`), see `results/explainability/<backbone>/`.
