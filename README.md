# The Threshold, Not the Score

## Selective Prediction Fails at Threshold Selection Under Distribution Shift

> A reproducible empirical study of where selective prediction breaks, why it breaks, and what labelled information is required to restore a statistically defensible risk guarantee.

[![Code](https://img.shields.io/badge/code-GitHub-black)](https://github.com/shayamahmad/threshold-distribution_shift)
[![Framework](https://img.shields.io/badge/framework-PyTorch-red)](https://pytorch.org/)
[![Data](https://img.shields.io/badge/data-CIFAR--10%20%7C%20CIFAR--10--C%20%7C%20CIFAR--10.1-blue)](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## Abstract

Selective prediction gives a classifier the option to abstain rather than answer every input. This is attractive whenever an error is more costly than an unanswered example. In practice, however, selective prediction requires a **threshold**: predictions above the threshold are accepted, predictions below it are rejected.

This repository investigates a deceptively simple but consequential question:

> If a threshold is calibrated to guarantee a target selective error rate on clean validation data, does that guarantee survive when the deployment distribution shifts?

The answer observed in this study is consistently negative across the evaluated settings.

For a target selective risk of **5%**, a threshold transferred from clean validation data reaches approximately **42.3% error at the strongest CIFAR-10-C corruption level** for the ResNet18 ensemble, while coverage stays around **89%**. For WideResNet-16-4, the corresponding error reaches **45.2%**. The failure is not explained by a complete collapse of the ranking induced by the confidence score: an oracle threshold applied to the same maximum-probability score can still achieve approximately the target risk at every corruption severity.

This motivates the central decomposition of the paper:

```
Selective prediction
  |
  |-- Ranking / sorting: which examples look safer?
  |
  |-- Threshold selection: where should the system stop accepting?
```

The experiments indicate that the second component is the critical failure point.

Four confidence scores produce very similar ranking quality, while the threshold transferred from clean data accepts substantially more examples than a threshold chosen with access to shifted labels. At the strongest corruption level, the clean-transfer rule accepts **67-77 percentage points more coverage** than is compatible with the 5% target, depending on model and corruption set.

The study then evaluates label-free threshold-selection baselines, held-out corruptions, a second architecture, CIFAR-10.1 natural distribution shift, and finite-sample label-based risk control. A statistically conservative procedure can restore the target guarantee, but the experiments expose a practical trade-off: under stronger shift, thousands of labelled deployment examples may be required to recover high fractions of the oracle-achievable coverage.

The contribution is deliberately **not** a new neural architecture. It is a controlled empirical and statistical diagnosis of a deployment failure mode that can otherwise be hidden behind calibration, accuracy, or uncertainty-quality metrics alone.

---

## 1. Why This Study Matters

Machine-learning systems are often evaluated on the assumption that the validation distribution represents the deployment distribution. That assumption is fragile — images change because of lighting, cameras, compression, acquisition protocols, environmental conditions, population changes, time, and sensor characteristics.

Selective prediction is often proposed as a safety mechanism:

```
Input
  |
  v
Classifier
  |
  |-- sufficiently confident --> Predict
  |
  |-- insufficiently confident --> Abstain
```

The apparent logic:

1. Calibrate confidence.
2. Choose a threshold corresponding to a desired error rate.
3. Reject low-confidence cases.
4. Deploy.

The critical question is whether step 2 remains valid after the distribution changes. A threshold is not merely a number — it is a claim about which region of the score distribution is safe enough to act on. If the relationship between score and correctness changes under shift, a threshold learned on the old distribution can stop representing the intended risk level.

This repository studies where the deployment guarantee actually fails.

---

## 2. Central Research Question

> When a confidence threshold is calibrated on a clean validation distribution, can it continue to guarantee a target selective-risk level under distribution shift? If it cannot, can labelled samples from the shifted distribution recover a statistically defensible guarantee, and what is the coverage cost of doing so?

This is more specific than asking whether accuracy decreases, calibration degrades, uncertainty scores worsen, or a model becomes overconfident under shift — those phenomena are already well studied. The distinctive focus here is the **threshold-selection layer** of a deployed selective system.

---

## 3. The Key Conceptual Decomposition

The paper separates selective prediction into two logically distinct operations.

### 3.1 Ranking

A score assigns each example a degree of apparent confidence: `g(x)`, a real number. Examples can then be sorted from highest to lowest score. This answers: **which examples does the model consider safer than which other examples?**

### 3.2 Stopping

A threshold `t` determines where the system stops accepting predictions:

```
A_t = { x : g(x) >= t }
```

This answers: **how many examples can the system safely accept?**

The distinction matters because a ranking can remain useful while the threshold calibrated on another distribution becomes invalid.

```
             SCORE / RANKING
                    |
                    v
        Which examples are safer?
                    |
                    v
              SORTED EXAMPLES
                    |
                    v
             THRESHOLD / STOPPING
                    |
                    v
          Which examples are
          actually accepted?
```

The experiments were designed to measure these components separately.

---

## 4. Selective Risk and Coverage

Let `g(x)` be a confidence score and `t` a threshold.

The system accepts: `A_t = { x : g(x) >= t }`

**Selective risk:**

```
R(t) = P( Y_hat != Y | X in A_t )
```

**Coverage:**

```
C(t) = P( X in A_t )
```

The deployment objective is `R(t) <= alpha`, while retaining as much coverage as possible. The main experiments use `alpha = 0.05`.

So the intended operational statement is: **among the predictions the system chooses to make, no more than approximately 5% should be wrong.** This is fundamentally different from simply reporting overall classification accuracy.

---

## 5. The Most Important Distinction: Empirical Performance vs Guarantee

A recurring problem in ML evaluation is treating an observed error rate as if it were a guarantee.

Suppose a shifted test sample produces an observed risk `R_hat = 0.04`. That does **not** automatically establish `R <= 0.05` for the deployment distribution — finite samples introduce uncertainty, threshold selection introduces another layer of statistical dependence, and multiple candidate thresholds introduce yet another problem.

```
Observed empirical risk
        !=
Statistically controlled risk
        !=
Deployment-time guarantee
```

This distinction is central to the later experiments.

---

## 6. Datasets

### 6.1 CIFAR-10

Provides the clean training and validation environment: 50,000 training images, 10,000 test images, 10 classes, 32x32 RGB images.

The training set is split once into 45,000 training / 5,000 validation examples using a fixed shuffle with seed `12345`. The same split is reused throughout. This matters because the clean validation distribution determines the transferred threshold.

### 6.2 CIFAR-10-C

Provides controlled distribution shift via 15 standard corruption families: Gaussian noise, shot noise, impulse noise, defocus blur, glass blur, motion blur, zoom blur, snow, frost, fog, brightness, contrast, elastic transform, pixelation, JPEG compression. Each has five severity levels.

```
Clean -> Severity 1 -> Severity 2 -> Severity 3 -> Severity 4 -> Severity 5
```

The experiments measure how the failure evolves as shift strengthens, not just clean vs. shifted.

### 6.3 Held-Out Corruptions

Four corruption types are deliberately withheld from method/threshold/metric selection: speckle noise, Gaussian blur, spatter, saturate. They are evaluated afterward as an internal holdout, to check the result isn't an artifact of tuning against the same 15 corruptions.

### 6.4 CIFAR-10.1

2,000 newly collected images intended to resemble the original CIFAR-10 distribution while coming from a later, independent collection process — a natural (non-synthetic) distribution shift test.

---

## 7. Models

Two architectures, each trained with three random seeds (final results use the three-seed ensemble):

- **ResNet18** — adapted for CIFAR-sized images (3x3 first convolution, stride 1, initial pooling removed)
- **WideResNet-16-4** — used to test whether the observed behavior is specific to one network design

```
2 architectures x 3 random seeds
```

---

## 8. Training and Calibration

- Stochastic gradient descent, Nesterov momentum 0.9
- Weight decay 5e-4
- Batch size 128
- One-cycle learning-rate schedule, peak LR 0.1
- 60 epochs
- Random crops (padding 4), horizontal flips
- Batch normalization, mixed precision

Temperature scaling is fitted to the ensemble on the clean validation set.

| Model | Temperature | Clean accuracy |
|---|---:|---:|
| ResNet18 | 1.242 | 95.3% |
| WideResNet-16-4 | 1.219 | 95.0% |

Temperatures above 1 indicate the models were overconfident before calibration.

---

## 9. Why the Saved Logits Matter for Reproducibility

Model outputs for every experimental condition are computed once and saved. Every downstream experiment reuses those same saved outputs. Differences between threshold-selection methods are therefore not caused by repeatedly retraining or rerunning the networks — the comparison stays concentrated on the actual object of study: **the selection procedure.** This keeps the pipeline easy to audit and cheap to reproduce.

---

## 10. Confidence Scores

Four scores are evaluated:

**Maximum Softmax Probability**
```
g_MSP(x) = max_c p_c
```

**Margin** (separation between top two classes)
```
g_margin(x) = p_(1) - p_(2)
```

**Negative Entropy** (larger = more concentrated prediction)
```
g_ent(x) = sum_c [ p_c * log(p_c) ]
```

**Ensemble Disagreement** (disagreement among the three ensemble members)
```
g_ens(x) = - sum_c Var_s[ p_c^(s) ]
```

The purpose is not to crown a single best score, but to test whether the threshold-transfer failure can be explained simply by a poor ranking score.

---

## 11. Threshold-Selection Methods

Five principal rules are compared:

1. **Clean Transfer** — threshold selected on clean validation data for `alpha = 0.05`, applied unchanged to shifted data. This is the deployment pattern under investigation.
2. **Mean Confidence** — accepted-set error approximated as `1 - mean confidence`; the largest coverage satisfying the target under this approximation is chosen. Intentionally simple, to test whether a lightweight label-free correction already helps.
3. **ATC-MC** — thresholded accuracy-estimation approach (Garg et al.) using maximum softmax probability.
4. **ATC-NE** — same idea, using negative entropy.
5. **Oracle** — has access to true labels of the shifted distribution; selects the threshold that actually achieves the target risk while maximizing coverage. Not deployable — a reference point for "how much coverage was actually achievable with the correct labels."

---

## 12. Result 1 — Confidence Degrades More Slowly Than Accuracy

Under corruption, accuracy falls much faster than average confidence. For one ResNet18 seed averaged across the standard CIFAR-10-C corruptions:

- Accuracy falls by approximately 39 percentage points from clean to severity 5
- Average confidence falls by approximately 12 percentage points

At the strongest severity, the model can be wrong on roughly 45% of inputs while still reporting approximately 85% average confidence.

```
Actual correctness:  down down down down down down down
Confidence:           down down
```

The confidence signal does not move enough to warn the thresholding system that the data have become substantially harder.

---

## 13. Result 2 — The Clean Threshold Fails Under Shift

For a 5% target, the clean threshold is transferred directly to CIFAR-10-C. ResNet18 ensemble:

| Corruption severity | Real error | Coverage |
|---:|---:|---:|
| 1 | 0.118 | 0.973 |
| 2 | 0.172 | 0.960 |
| 3 | 0.233 | 0.944 |
| 4 | 0.306 | 0.923 |
| 5 | 0.423 | 0.890 |

At severity 5: `R ≈ 42.3%` against a target of `5%`. Risk ratio: `0.423 / 0.05 = 8.46`.

The system does not become conservative enough — it continues accepting almost all examples while the correctness of accepted examples deteriorates sharply.

For WideResNet-16-4, the strongest-corruption error reaches approximately **45.2%**. The worst individual corruption reported (impulse noise) reaches approximately **74.5%**.

---

## 14. Result 3 — Stricter Guarantees Fail by Larger Multipliers

At severity 5, ResNet18:

| Target risk | Real error | Risk ratio | Coverage |
|---:|---:|---:|---:|
| 10% | 44.3% | 4.4x | 98.6% |
| 5% | 42.0% | 8.4x | 89.0% |
| 2% | 28.8% | 14.4x | 52.0% |
| 1% | 19.1% | 19.1x | 36.2% |

A stricter target does reduce the absolute observed error, because it raises the clean-data threshold — but the **relative failure grows**. At a 1% target, the deployed error is roughly nineteen times the promised level. Reporting only absolute error can hide how serious the failure is.

---

## 15. Result 4 — The Target Is Not Impossible

If the target itself were impossible under shift, the clean-threshold failure wouldn't tell us much about threshold selection. So an oracle threshold is computed using the shifted labels. ResNet18 ensemble, 5% target:

| Severity | Oracle coverage | Oracle error |
|---:|---:|---:|
| 0 | 1.000 | 0.047 |
| 1 | 0.833 | 0.050 |
| 2 | 0.696 | 0.050 |
| 3 | 0.584 | 0.050 |
| 4 | 0.458 | 0.050 |
| 5 | 0.264 | 0.049 |

At severity 5:

```
Clean-transfer coverage  ≈ 99.5%
Oracle-achievable coverage ≈ 26.4%
```

The target **can** be met. The system simply chooses the wrong stopping point. This is the empirical basis for the central claim: **the ranking signal remains sufficiently useful to support the target, but the threshold transferred from the clean distribution does not know where to stop.**

---

## 16. Result 5 — The Ranking Is Not the Main Bottleneck

If threshold transfer fails because maximum softmax probability is a poor ranking score, swapping the score should substantially improve oracle-achievable coverage. AURC by score and severity:

| Score | Severity 1 | Severity 3 | Severity 5 |
|---|---:|---:|---:|
| Maximum probability | 32.1 | 96.3 | 255.6 |
| Margin | 32.3 | 97.3 | 257.6 |
| Negative entropy | 32.3 | 96.2 | 254.7 |
| Model disagreement | 34.8 | 103.3 | 269.1 |

Maximum probability, margin, and negative entropy stay within roughly 1% of each other across the reported severities. Model disagreement is somewhat worse. Maximum probability and margin achieve essentially identical target-achievable coverage to three decimal places.

So the experiments do **not** support "the threshold failed because the score stopped ranking usefully." They support: **the ranking remains useful, but the clean-distribution mapping from score to acceptable stopping point does not transfer.**

---

## 17. Result 6 — Coverage Error as a Diagnostic

**Coverage error** = the amount of coverage a method selects beyond the coverage achievable at the target. Positive coverage error means the method is accepting examples that should have been rejected to hold the target risk.

Severity 5, 5% target:

| Threshold rule | ResNet18 standard | ResNet18 held-out | WideResNet standard | WideResNet held-out |
|---|---:|---:|---:|---:|
| Clean transfer | +0.731 | +0.675 | +0.769 | +0.692 |
| ATC-MC | +0.415 | +0.406 | +0.469 | +0.427 |
| ATC-NE | +0.311 | +0.310 | +0.330 | +0.298 |
| Mean confidence | +0.227 | +0.228 | +0.170 | +0.155 |
| Oracle | 0.000 | 0.000 | 0.000 | 0.000 |

The same ordering holds across architecture, corruption set, and held-out corruption types — an important internal robustness check.

---

## 18. Result 7 — Held-Out Corruptions Replicate the Failure

At severity 5: clean transfer reaches approximately **0.386** coverage error for ResNet18 and **0.421** for WideResNet-16-4, on corruptions never used to choose methods or settings. The shape and ordering of methods stays similar to the standard corruption set, making it unlikely the central result is an artifact of repeatedly examining the same 15 corruptions.

---

## 19. Result 8 — Natural Distribution Shift Produces the Same Problem

CIFAR-10.1 accuracy reduction from clean: approximately 6.4 points (ResNet18), 6.9 points (WideResNet-16-4). The transferred 5% threshold nevertheless produces:

| Model | Error | 95% bootstrap range | Risk ratio | Coverage | Oracle coverage |
|---|---:|---|---:|---:|---:|
| ResNet18 | 0.111 | [0.097, 0.125] | 2.2x | 1.000 | 0.828 |
| WideResNet-16-4 | 0.119 | [0.105, 0.133] | 2.4x | 1.000 | 0.845 |

```
The system answers everything.
The target is 5%.
The observed error is ~11-12%.
```

Yet an oracle threshold could still hit the target at more than 80% coverage. The phenomenon is not confined to severe synthetic corruption.

---

## 20. Result 9 — Label-Free Correction Helps, But Doesn't Restore the Guarantee

At severity 5, the ordering is consistently:

```
Oracle > Mean confidence > ATC-NE > ATC-MC > Clean transfer
```

The simple mean-confidence rule beats both evaluated ATC variants in the reported settings. None restores the formal target guarantee. **Label-free heuristics reduce over-acceptance, but reducing coverage error is not the same as obtaining a statistically valid deployment guarantee.**

---

## 21. Result 10 — A Statistical Guarantee Requires Deployment Information

The final method uses a labelled sample from the shifted distribution:

1. Fix a candidate threshold grid using clean validation information (not deployment labels).
2. Evaluate candidate thresholds using a labelled deployment sample.
3. Compute an exact binomial upper confidence bound.
4. Apply a multiple-testing correction.
5. Search from loose to strict thresholds.
6. Return the first threshold whose bound satisfies the target.
7. Abstain on everything if no candidate can be supported.

The question shifts from "which threshold looks best?" to "which threshold can I statistically justify from the labels I have?"

---

## 22. Why Earlier Risk-Control Procedures Were Rejected

### 22.1 The 100-Point Bonferroni Procedure

100 candidate thresholds with a Bonferroni correction. Safe, but impractical — with 100 labels or fewer it could stay silent on everything. Reported real error was around 0.008 against a 0.05 target: substantial unused statistical slack.

```
More conservative correction -> stronger statistical protection -> lower practical coverage
```

### 22.2 Fixed-Sequence Testing

Walked thresholds in the opposite direction using fixed-sequence testing. Produced an unacceptable pattern:

```
500 labels  -> 0.57 coverage
4000 labels -> 0.20 coverage
```

Coverage should not deteriorate as more deployment labels become available. Diagnosis: the earliest tests in the sequence were evaluated at extremely strict thresholds where too few labelled examples were accepted to give enough statistical evidence — the procedure stalled from sample scarcity, not because the underlying threshold was actually inappropriate. Lesson: a theoretically valid testing framework can still be operationally inappropriate when hypothesis ordering interacts badly with sparse accepted samples.

---

## 23. Final Risk-Control Procedure

- 20 candidate thresholds
- Candidates derived ahead of time from validation confidence quantiles
- Exact binomial upper bound
- Bonferroni-adjusted confidence level
- Loose-to-strict threshold ordering
- Target risk 5%, overall failure allowance 10%

Intentionally conservative, and that conservatism is acknowledged rather than hidden. It passes the study's monotonicity check: **coverage increases as the deployment label budget increases.**

---

## 24. How Many Labels Does a Guarantee Cost?

Efficiency = achieved coverage / oracle coverage (1.0 = full oracle-achievable coverage recovered).

**CIFAR-10-C**

| Setting | 250 | 500 | 1000 | 2000 | 4000 |
|---|---:|---:|---:|---:|---:|
| ResNet18, severity 3 | 0.39 | 0.64 | 0.76 | 0.82 | 0.85 |
| ResNet18, severity 5 | 0.18 | 0.32 | 0.52 | 0.67 | 0.78 |
| WideResNet, severity 3 | 0.33 | 0.59 | 0.74 | 0.80 | 0.86 |
| WideResNet, severity 5 | 0.19 | 0.29 | 0.46 | 0.62 | 0.71 |

Approximately 2,000 labels are needed to reach roughly 80% efficiency at severity 3; more than 4,000 may be needed at severity 5. The target is violated in only about 0-1% of runs, against an allowed failure level of 10%.

**CIFAR-10.1**

| Setting | 200 | 400 | 800 | 1200 |
|---|---:|---:|---:|---:|
| ResNet18 error | 0.007 | 0.015 | 0.023 | 0.028 |
| ResNet18 efficiency | 0.27 | 0.68 | 0.83 | 0.88 |
| WideResNet efficiency | 0.19 | 0.67 | 0.79 | 0.80 |

Roughly 800 labels reach around 80% efficiency in the reported ResNet18 setting. The cost of recovering a guarantee depends strongly on the severity and geometry of the distribution shift.

---

## 25. Why a Simple Label-Count Calculation Underestimates the Cost

A tempting shortcut: "if I only need to demonstrate 5% error, I only need enough accepted examples to make the probability of observing zero errors sufficiently small." That reasoning predicts roughly 178 labels in one configuration. The observed requirement is closer to approximately 2,000.

Why: the true error at the oracle threshold is approximately equal to the target. If `R = alpha`, finite data cannot reliably prove `R <= alpha` without statistical uncertainty forcing the procedure toward a safer threshold. The method needs enough labels not just to observe zero errors, but to distinguish "risk ≈ target" from "risk safely below target" — which is why the empirical label requirement is much larger than a naive zero-error sample-size calculation suggests.

```
Stronger distribution shift
        -> Lower oracle-achievable coverage
        -> More difficult threshold identification
        -> More labelled deployment data required
```

The systems that most need reliable deployment-time risk control can be the systems for which the required labels are hardest to obtain — a reason to measure the label cost before claiming a guarantee, not a reason to abandon selective prediction.

---

## 26. Class-Level Analysis

A single aggregate risk number can hide class differences. At severity 3 with the clean-transfer threshold, accepted-set error varies from about **13.3%** (frog) to **42.3%** (dog) — roughly a 3.2x spread — while coverage stays near 99% across classes. The system does not necessarily abstain more on the classes where its predictions have become less reliable, which is an additional reason to look beyond one global accuracy or confidence number.

---

## 27. What This Paper Does *Not* Claim

It does not claim: that all confidence scores fail under all forms of shift; that CIFAR-10-C represents every real-world distribution shift; that the reported label budgets are universal minimum requirements; that the final risk-control procedure is optimal; that the method should replace all calibration techniques; that a particular uncertainty score is universally inferior; or that one statistical correction is universally best.

> For the evaluated CIFAR-scale image-classification setting, a confidence threshold calibrated on clean validation data can lose its selective-risk target under distribution shift even when the underlying ranking remains sufficiently useful to achieve that target with an appropriately selected threshold.

That narrower claim is both testable and falsifiable.

---

## 28. Why the Negative Results Matter

Preserving the rejected methods avoids a misleadingly simple story ("we tried one method, invented a better one, and it worked"). The actual development path:

```
Baseline transfer
  -> Failure observed
  -> Is the score broken?
  -> Oracle says target remains achievable
  -> Multiple scores say ranking is similar
  -> Can label-free threshold correction work?
  -> Partial improvement, no guarantee
  -> Can statistical risk control work?
  -> First method too conservative
  -> Second method structurally misordered
  -> Final method passes validity and monotonicity checks
  -> Label cost quantified
```

This makes the repository a reproducibility record of scientific reasoning, not just a collection of successful outputs.

---

## 29. Threats to Validity

- **Dataset scope** — CIFAR-sized image classification; results shouldn't be generalized automatically to ImageNet-scale models, language models, multimodal models, medical datasets, robotics, or other domains without separate validation.
- **Model scope** — only two architecture families; replication across architectures, not universality.
- **Training regime** — models reach very low training loss and may exhibit strong confidence; different regularization, early stopping, augmentation, or architecture could change the numerical magnitude. The *direction* of the threshold-transfer problem is the object of interest, not the claim that the numbers are universal constants.
- **CIFAR-10.1 size** — only 2,000 images, so large label budgets consume a substantial fraction of available evaluation data. The 200/400-label settings are practical demonstrations; larger budgets mainly reveal the shape of the efficiency curve.
- **Final procedure is conservative** — achieves approximately 0.017 real error against a 0.050 target, meaning it is not statistically or computationally optimal. Reported label requirements should be read as upper-bound-like practical costs for *this* procedure, not fundamental lower bounds for every possible method.

---

## 30. What Would Falsify the Main Interpretation?

The claim that threshold selection is the main bottleneck would be weakened if a substantially better confidence score materially improved ranking metrics, substantially increased oracle-achievable coverage, *and* simultaneously restored the transferred clean threshold. The deployment-label interpretation would be weakened if a genuinely label-free method could reliably recover the 5% guarantee under the same shifts without unverifiable assumptions. These criteria distinguish the paper's claim from a vague "models become uncertain under shift."

---

## 31. Reviewer-Facing Summary of the Contribution

**A.** Identifies a specific failure layer — not "confidence gets worse" but "the threshold selected on clean validation data fails to transfer, even when the score remains capable of supporting the target under a correctly selected threshold."

**B.** Separates ranking from stopping, via oracle and multi-score experiments.

**C.** Quantifies the failure — risk ratios, coverage, coverage error, oracle coverage, AURC, bootstrap intervals, label-budget efficiency.

**D.** Evaluates negative results — rejected statistical procedures are retained and explained.

**E.** Evaluates unseen corruptions — reduces the chance the result is a product of repeated tuning.

**F.** Evaluates natural distribution shift — CIFAR-10.1 shows the issue isn't restricted to artificial corruption.

**G.** Quantifies the cost of fixing the problem — moves from "the guarantee fails" to "how many labels are required to obtain a statistically defensible replacement?"

---

## 32. Practical Implication

> A confidence threshold calibrated on clean validation data should not be described as enforcing a target selective-risk level after distribution shift unless the transfer assumption is justified or the deployment distribution is explicitly accounted for.

A deployed system has several options, and the paper quantifies the trade-off rather than prescribing one:

1. Accept the shift and make no formal risk guarantee.
2. Obtain labelled deployment data and recalibrate / select a threshold.
3. Use a method with a formal guarantee under clearly stated assumptions.
4. Abstain more aggressively and accept lower coverage.

---

## 33. Broader Relevance

Any system combining `prediction + confidence score + accept/reject threshold` can encounter the same conceptual issue: medical image triage, automated quality inspection, scientific image analysis, remote sensing, fraud screening, document classification, safety monitoring, and other selective-decision systems. Domain-specific numbers will differ; the statistical question stays the same: **does the deployment threshold still support the claim it was calibrated to support?**

---

## 34. Repository Structure

```
paper.tex
paper.pdf
README.md

notebooks/
    phase-1-sep-14.ipynb
    phase-2-sep-15.ipynb
    phase-3-sep-15.ipynb
    phase-4-sep-15.ipynb
    phase-5-sep-15.ipynb
    phase-6-sep-15.ipynb
    phase-7-sep-15.ipynb
    phase-8-sep-15.ipynb
    phase-9-sep-15.ipynb
    phase-10-sep-15.ipynb
    phase-11-sep-15.ipynb

figures/
    fig1.pdf
    fig2.pdf
    fig3.pdf
    fig4.pdf
    fig5.pdf
    make_figures.py

fig1.pdf
fig2.pdf
fig3.pdf
fig4.pdf
fig5.pdf

sn-jnl.cls
sn-mathphys-num.bst
```

---

## 35. Experimental Pipeline

| Phase | Description | Status |
|---|---|---|
| 1 | Train ResNet18 (3 seeds), fetch CIFAR-10-C, save model outputs | — |
| 2 | Temperature scaling; initial selective-prediction baseline | — |
| 3 | Sweep target levels; evaluate oracle-achievable coverage | — |
| 4 | ATC / Difference-of-Confidences baselines; first label-budget experiments | — |
| 5 | Compare coverage error and AURC across confidence scores | — |
| 6 | Train WideResNet-16-4 (3 seeds); add withheld corruption set | — |
| 7 | Replicate key observations across architecture and corruption set | — |
| 8 | Evaluate 100-point Bonferroni risk-control procedure | Rejected — too conservative |
| 9 | Evaluate fixed-sequence testing | Rejected — threshold-ordering / sparse-acceptance problem |
| 10 | Evaluate CIFAR-10.1 | — |
| 11 | Evaluate final 20-point loose-to-strict risk-control procedure | Reported |

**Reproduction order:**

```
phase-1 -> phase-2 -> phase-3 -> phase-4 -> phase-5 -> phase-6
        -> phase-7 -> phase-8 -> phase-9 -> phase-10 -> phase-11
```

Phases 2-5 use Phase 1 outputs. Phases 7-11 use both Phase 1 and Phase 6 outputs. Phase 1 and Phase 6 require GPU resources; the rest are primarily CPU-based once model outputs are saved.

---

## 36. Computational Cost

- 1x NVIDIA T4
- 6 trained models
- Approximately 65 minutes total training

Once model outputs are saved, subsequent experiments operate on stored predictions rather than repeatedly executing the neural networks — the statistical experiments are cheap to rerun and inspect.

---

## 37. Statistical Reporting

- Bootstrap uncertainty: 200 resamples for CIFAR-10-C, 2,000 resamples for CIFAR-10.1
- Label-budget experiments: 20 draws per corruption for CIFAR-10-C, 200 draws for CIFAR-10.1
- Threshold selection and evaluation are separated so that the same sampled points are never used for both operations in the label-budget experiments — essential for avoiding an artificially optimistic evaluation

---

## 38. Data Availability

Publicly available datasets only — no private or proprietary data required:

- CIFAR-10
- CIFAR-10-C
- CIFAR-10.1 (version 6)

The notebooks download the datasets directly.

---

## 39. Code Availability

**https://github.com/shayamahmad/threshold-distribution_shift**

---

## 40. Build the Manuscript

The repository includes the Springer Nature class and bibliography style.

```bash
pdflatex paper.tex
pdflatex paper.tex
```

Or upload the repository contents to Overleaf and use `paper.tex` as the main document.

---

## 41. Reproducing the Experiments

```bash
git clone https://github.com/shayamahmad/threshold-distribution_shift.git
cd threshold-distribution_shift
```

Create an environment appropriate for the notebooks:

```bash
pip install numpy pandas scipy scikit-learn
pip install matplotlib seaborn
pip install torch torchvision
pip install jupyter
```

Then:

```bash
jupyter notebook
```

Run the notebooks in the documented phase order. For the training phases, use the GPU configuration described in the manuscript.

---

## 42. Reproducibility Checklist

- [ ] CIFAR-10 split is identical to the study split
- [ ] Random seeds are preserved
- [ ] Model outputs correspond to the intended architecture and seed
- [ ] Temperature scaling is fitted only on clean validation data
- [ ] CIFAR-10-C severity blocks are separated correctly
- [ ] Held-out corruptions are not used for method selection
- [ ] Threshold selection does not use evaluation labels
- [ ] Label-selection and evaluation samples are disjoint
- [ ] Candidate thresholds are fixed independently of the deployment labels
- [ ] Statistical corrections are applied to the intended hypothesis family
- [ ] Zero-coverage outcomes are retained rather than discarded
- [ ] Bootstrap intervals use the specified resampling scheme
- [ ] Numerical claims match the manuscript tables and figures

---

## 43. Publication Case in One Page

| Step | Summary |
|---|---|
| Problem | Selective prediction is meant to let models abstain when likely wrong |
| Hidden assumption | A clean-calibrated threshold is often treated as still meaningful after deployment shift |
| Controlled experiment | Transfer a clean 5% threshold to progressively shifted CIFAR-10-C data |
| Observation | At maximum corruption, ResNet18 ensemble reaches ~42.3% error at ~89% coverage |
| Diagnosis | An oracle threshold on the same score can still achieve ~5% risk |
| Ranking test | Four scores have very similar ranking performance |
| Interpretation | The failure is concentrated in threshold selection, not example ordering |
| Robustness check | Same method ordering across 2 architectures, held-out corruptions, and CIFAR-10.1 |
| Practical question | Can a label-free method recover the guarantee? |
| Result | Label-free heuristics improve coverage error but don't restore a formal guarantee |
| Statistical solution | A finite-sample label-based procedure can recover the target |
| Deployment cost | Required labels rise substantially as shift strengthens |

---

## 44. The Core Scientific Message

> **A good confidence ranking is not the same thing as a valid deployment threshold.**

Under distribution shift, a model can still rank examples reasonably well while the threshold calibrated on clean data becomes dangerously permissive. The study demonstrates this by transferring the clean threshold, measuring the resulting risk, constructing an oracle threshold, comparing multiple scores, evaluating label-free selectors, testing held-out corruptions, testing natural distribution shift, implementing finite-sample risk control, exposing failed statistical constructions, and quantifying the deployment label budget.

Not simply "models become worse under shift," but more specifically: **the deployment failure can occur at the threshold-selection layer even when the score retains enough ranking information to support the desired selective-risk target.**

---

## 45. Final Perspective

A selective prediction system makes an implicit promise: *"I will answer only when the prediction is safe enough."* This study asks whether that promise survives when the world changes. In the evaluated setting, the answer is not guaranteed merely because the threshold worked on clean validation data.

```
Do not ask only:      "How good is the confidence score?"
Also ask:              "Where should the threshold be?"
Then ask:               "Can that threshold be justified after distribution shift?"
Finally ask:            "How much deployment information is required to justify it?"
```

That sequence is the conceptual contribution of the repository.

---

## 46. Citation

```bibtex
@article{ahmad_threshold_distribution_shift,
  title   = {The Threshold, Not the Score: Selective Prediction Fails at Threshold Selection Under Distribution Shift},
  author  = {Ahmad, Shayam},
  year    = {2026},
  note    = {Research manuscript}
}
```

---

## 47. License

Add the appropriate open-source license before public release.

---

## Repository Status

**Research manuscript and reproducibility repository.**

The repository contains the manuscript source, figures, experimental notebooks, and methodological development used in the study. The reported conclusions are intentionally scoped to the evaluated datasets, architectures, distribution shifts, thresholds, and statistical procedures. Extending the conclusions beyond these settings should be treated as a hypothesis requiring additional empirical validation.

---

### One-Sentence Research Question

> When distribution shift changes the relationship between confidence and correctness, can a selective classifier still make a statistically defensible risk promise, and what labelled information is required to choose the threshold that keeps that promise?
