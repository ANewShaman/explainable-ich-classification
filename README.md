# Intracranial Hemorrhage Detection and Explainability

## Overview
Forensic pathology teams are short on specialists and post-mortem CT (PMCT) is expensive to annotate at scale, while clinical CT archives are not. The RSNA Intracranial Hemorrhage (ICH) Detection Challenge dataset offers hundreds of thousands of labeled slices, which makes it a practical foundation for training ICH classifiers before anything is attempted on PMCT. But a model that classifies well isn't automatically a model a forensic reviewer can trust. This project builds three ICH classifiers on RSNA data and asks a second question alongside classification: are their explanations faithful and stable enough to be useful, without relying on expensive pixel-level annotations?

## Research Question
Can a clinically trained ICH detection model produce explanations that are faithful, stable, and consistent across architectures, well enough to support a forensic reviewer working with post-mortem CT, without needing pixel-level ground truth?

A narrower sub-question drives the actual experiments: given three backbone and training configurations evaluated on RSNA, which one provides the strongest evidence across classification performance, explanation faithfulness (AOPC), explanation stability (Max-Sensitivity), calibration, and a bone-window correlation check? No aggregation method was defined ahead of time, so there's no single combined score — the architectures are compared metric by metric instead.

## Dataset
- **RSNA ICH CT dataset** — the RSNA Intracranial Hemorrhage Detection Challenge set, with five hemorrhage subtypes (epidural, intraparenchymal, intraventricular, subarachnoid, subdural) plus an aggregate "any" label.
- **Split** — 70/15/15 (train/val/test), stratified on the binary "any" label, `random_state=42`. Train: 188,402 slices. Validation: 40,372. Test: 40,373. The validation set was used for early stopping, per-class F1-optimal thresholds, and temperature-scaling calibration; the test set was untouched during tuning and used only for reported metrics.
- **Input representation** — each CT slice is rendered as a 3-channel image using three clinically meaningful CT windows, not a cosmetic color map: brain window (W80/L40) for soft-tissue contrast, subdural window (W200/L80) for wider blood/brain contrast, and bone window (W2000/L600) for skull detail. ImageNet normalization is applied throughout; training uses random horizontal flips, evaluation does not.
- **Sampling** — a `BalancedRandomSampler` equalizes the sampling probability of any=1 vs any=0 per training epoch. It does not rebalance the five subtypes conditional on any=1, so epidural (≈1.18% test prevalence) stays the rarest positive class — which matters a lot for how its metrics read later.

Because the bone window is one of the three input channels, a Grad-CAM map could in principle latch onto skull intensity rather than the hemorrhage itself. That possibility is exactly why the bone-window correlation check exists (see Explainability & Reliability Analysis below).

## Methodology
### Models
Three architecturally distinct backbones, all ImageNet-pretrained with most layers frozen, fine-tuned with a new `Dropout(0.3) → Linear` head:

| | ResNet-50 | ViT-B/16 | ConvNeXt-Base |
|---|---|---|---|
| Unfrozen depth | layer4 | final encoder block only | last stage only |
| Optimizer | SGD (momentum 0.9) | AdamW | AdamW |
| LR | 1e-3 | 1e-4 | 1e-4 |
| Warmup | none | 1 epoch linear | 1 epoch linear |
| Scheduler | cosine annealing | cosine annealing after warmup | cosine annealing after warmup |
| Best checkpoint | epoch 14 (val loss 0.1905) | epoch 5 (val loss 0.1726) | epoch 13 (val loss 0.1759) |

These are three complete, reasonable training setups rather than a controlled ablation — optimizer, LR, warmup, and fine-tuned depth all differ, so any explanation-quality difference can't be pinned on architecture alone.

### Evaluation
- **ROC-AUC / PR-AUC** — PR-AUC is reported alongside ROC-AUC because it's far more informative under class imbalance, and epidural hemorrhage is rare enough that ROC-AUC alone would be misleading.
- **F1** — computed at a per-class, validation-derived threshold.
- **Calibration** — temperature scaling fit on validation, evaluated on test, reported via ECE, MCE, Brier score, and NLL, both in aggregate and per subtype (a global temperature can help overall while quietly hurting a rare subtype).
- **Grad-CAM** — computed via `pytorch_grad_cam`, using plain Grad-CAM (not HiResCAM) across all three backbones for consistency. Target layers: `layer4[-1]` (ResNet-50), `features[-1][-1]` (ConvNeXt-Base), `encoder.layers[-1].ln_1` (ViT-B/16, with CLS token removed and the remaining 196 patch tokens reshaped into a 14×14 grid).
- **AOPC** (Area Over the Perturbation Curve) — tests whether the regions Grad-CAM ranks highly actually matter to the model's output, by cumulatively perturbing the top-ranked patches (16×16 grid, mean-fill per channel, 10 steps) and measuring the drop in predicted probability. Computed on 50 confirmed true positives per class.
- **Max-Sensitivity** — tests whether the Grad-CAM map itself is stable: small Gaussian noise is added to the input (r = 0.05σ, 10 draws), Grad-CAM is recomputed each time, and the maximum change in the map is reported. Lower is more stable.
- **Bone-window correlation check** — a Pearson correlation between each Grad-CAM map and the bone-window channel, to flag (not prove) possible reliance on skull intensity rather than the hemorrhage itself.

## Results
Macro-averaged classification performance on the held-out test set:

| Model | Macro ROC-AUC | Macro PR-AUC |
|---|---|---|
| ResNet-50 | 0.9186 | 0.6284 |
| ViT-B/16 | 0.9423 | 0.7180 |
| ConvNeXt-Base | 0.9403 | 0.6828 |

Epidural hemorrhage — the rarest class (1.18% test prevalence) — is where the architectures separate most clearly:

| Metric | ResNet-50 | ViT-B/16 | ConvNeXt-Base |
|---|---|---|---|
| ROC-AUC | 0.8703 | 0.9361 | 0.9291 |
| PR-AUC | 0.0944 | 0.3970 | 0.2513 |
| F1 | 0.1764 | 0.4144 | 0.3136 |
| AOPC | 0.114 ± 0.059 | 0.505 ± 0.206 | 0.338 ± 0.138 |
| Max-Sensitivity | 32.60 | 29.94 | 18.22 |

No single backbone wins everywhere. ViT-B/16 leads on aggregate classification, epidural AOPC, and aggregate calibration improvement. ConvNeXt-Base leads on intraparenchymal and intraventricular AOPC and has the lowest Max-Sensitivity across all six labels. ResNet-50 is clearly the weakest of the three on epidural classification and AOPC, without a large enough compensating advantage elsewhere.

## Explainability & Reliability Analysis
Classification accuracy doesn't tell you whether a model's explanation would actually help a reviewer, so three checks were layered on top of the standard metrics. AOPC asks whether the pixels Grad-CAM highlights actually drive the prediction — a model can lean heavily on a region for the wrong reason, so a good AOPC score is evidence of dependence, not evidence of anatomical correctness. Max-Sensitivity asks a different question: does the explanation map itself hold steady under small input perturbations, since an explanation that flips around under noise isn't one a reviewer can rely on. The bone-window check exists because the bone channel is literally part of the input — if a Grad-CAM map correlates strongly with skull intensity, that's a reason to look closer, though a low correlation doesn't rule out subtler shortcut behavior either.

Put together, the three checks disagree with each other in informative ways. ViT-B/16 has the strongest epidural AOPC, but several of its Grad-CAM maps are visually diffuse or drift toward peripheral skull margins, and its bone-correlation results show much larger case-to-case variation than the other two backbones (17/30 epidural cases exceed |r| = 0.2). ConvNeXt-Base's maps concentrate more tightly around plausible hemorrhage regions and its bone-correlation spread stays tighter. ResNet-50 sits in between on the bone check but is clearly behind on explanation faithfulness for the rarest class. The takeaway isn't "trust ConvNeXt-Base over ViT-B/16" — it's that perturbation sensitivity, visual plausibility, and shortcut-correlation don't always point the same direction, so all three need to be checked rather than assumed from classification performance alone.

**Status:** population-level statistical analysis for AOPC and Max-Sensitivity (confidence intervals, paired comparisons across backbones rather than point-estimate differences) is in progress. The n = 50 / n = 30 per-class results above are the point estimates reported in the manuscript.

## Limitations
- Not a controlled ablation — optimizer, learning rate, warmup, and fine-tuned depth differ across the three backbones, so differences can't be attributed to architecture alone.
- No ground-truth localization masks exist for RSNA, so Grad-CAM's anatomical correctness was never directly checked, only inferred indirectly.
- Grad-CAM cannot be treated as anatomically validated — AOPC and Max-Sensitivity describe model dependence and map stability, not whether the highlighted region is actually the hemorrhage.
- The bone-correlation check is a spatial-association sanity check, not causal proof. A high correlation doesn't confirm a bone shortcut, and a low one doesn't rule out a more indirect one.
- No statistical inference is reported yet — all comparisons are point estimates from n = 50 (AOPC, Max-Sensitivity) or n = 30 (bone correlation) per class, and the reported standard deviations show enough case-level spread that close means shouldn't be read as settled rankings.
- Independence across slices from the same patient is unresolved, so the effective amount of independent evidence may be smaller than the raw slice counts suggest.

## Future Work
- Domain-shift evaluation — clinical → forensic/PMCT transfer, carrying ViT-B/16 and ConvNeXt-Base forward as complementary candidates rather than picking one now.
- PMCT introduces artifacts absent from clinical CT (hypostasis, loss of grey-white differentiation, putrefactive gas, cerebral collapse, later liquefaction), so whether explanation faithfulness and stability survive that shift is the open question, not whether classification transfers — prior work already shows clinical-to-PMCT classification transfer is feasible.
- Add statistical rigor at that stage: confidence intervals for classification metrics, bootstrap intervals for AOPC/Max-Sensitivity, formal backbone comparisons instead of point-estimate rankings, and case-level (not slice-level) independence handling.

## Repository Structure
```
├── data/          # preprocessed dataset + README describing preprocessing
├── notebooks/      # training and evaluation notebooks per backbone
├── results/        # metrics, calibration outputs, explainability results (AOPC, Max-Sensitivity, bone-correlation, Grad-CAM figures) per model
├── LICENSE
└── README.md
```

## Reproducibility
- **Environment**: PyTorch, `pytorch_grad_cam` for explainability, standard ImageNet-pretrained weights (`ResNet50_Weights.IMAGENET1K_V2`, `ViT_B_16_Weights.IMAGENET1K_V1`, ConvNeXt-Base ImageNet1K weights).
- **Training**: see `notebooks/` for per-backbone training scripts; all three use `BCEWithLogitsLoss`, batch size 32, max 15 epochs with patience 5, and the training/val/test split fixed at `random_state=42`.
- **Evaluation**: thresholds are F1-optimal per class, selected on validation and applied to the held-out test set. Calibration (temperature scaling) is fit on validation and evaluated on test.
- **Explainability**: Grad-CAM, AOPC, and Max-Sensitivity are computed on confirmed true positives sampled without replacement (`random_state=42`); see `results/explainability/<backbone>/` for per-model outputs and figures.
