# Selective Prediction Under Distribution Shift

## Overview

This repository studies whether confidence thresholds calibrated on a clean validation distribution can continue to provide reliable risk guarantees when a classifier is deployed on a shifted data distribution.

The central question is:

> **Can a confidence threshold selected on clean validation data guarantee a target error rate after distribution shift? If not, how much labelled data from the shifted distribution is required to recover a statistically valid guarantee with useful coverage?**

The project focuses on **selective classification**, where a model may either make a prediction or abstain when it is not sufficiently confident.

---

## Research Motivation

In safety-sensitive or high-stakes machine learning applications, it is often desirable to accept only predictions for which the model is sufficiently confident.

For a target risk level of 5%, for example, we would like the accepted predictions to have an error rate no greater than 5%.

However, a threshold calibrated on clean validation data may not retain this property after deployment if the input distribution changes.

This project investigates that problem systematically.

---

## Key Concepts

### Selective Prediction

For every input, the classifier can:

- **Accept** the prediction when its confidence exceeds a threshold.
- **Abstain** when confidence is below the threshold.

Two quantities are especially important:

- **Risk:** error rate among accepted predictions.
- **Coverage:** fraction of all samples for which the model makes a prediction.

For example, if 800 of 1,000 samples are accepted and 40 accepted predictions are incorrect:

```text
Coverage = 800 / 1000 = 80%

Risk = 40 / 800 = 5%
```

The goal is therefore to maintain the desired risk while retaining as much coverage as possible.

---

## Experimental Setup

### Datasets

The experiments primarily use:

- **CIFAR-10** for clean training and validation.
- **CIFAR-10-C** for controlled distribution shifts caused by common image corruptions.
- **CIFAR-10.1** as an additional evaluation set representing a naturally collected distribution shift.

CIFAR-10-C includes corruptions such as:

- Gaussian noise
- Shot noise
- Blur
- Fog
- Snow
- Frost
- Brightness changes
- Contrast changes
- Pixelation
- JPEG compression

The corruption severity is varied to study progressively stronger distribution shifts.

---

## Models

The experiments include multiple image classification architectures, including:

- **ResNet-18**
- **WideResNet-16-4**

Multiple random seeds are used where applicable to reduce dependence on a single training run.

---

## Calibration

The clean validation distribution is used to calibrate model confidence.

The experiments include **temperature scaling**, which adjusts the model's confidence scores without changing its predicted class.

A confidence threshold is then selected for a target risk level, such as:

```text
Target Risk = 5%
```

The key question is whether this threshold remains valid when evaluated on shifted data.

---

## Experimental Questions

The project investigates several related questions.

### 1. Does clean-data calibration transfer under distribution shift?

A threshold selected using clean CIFAR-10 data is evaluated on shifted datasets such as CIFAR-10-C.

The experiment measures whether the accepted predictions continue to satisfy the target risk.

### 2. Does model confidence remain reliable after shift?

The experiments examine the relationship between confidence, coverage, and actual error under increasing distribution shift.

A key phenomenon investigated is that a model may remain highly confident while its actual error rate increases.

### 3. Can label-free threshold-selection methods solve the problem?

Several approaches are evaluated against an oracle procedure that has access to labels from the shifted distribution.

The objective is to determine how closely practical methods can approach the achievable coverage while maintaining the target risk.

### 4. How much labelled deployment data is required?

The later experiments introduce labelled examples from the shifted distribution.

The goal is to determine whether a relatively small label budget can provide statistically valid risk control without requiring full supervision of the deployment distribution.

### 5. Does statistical risk control remain valid after accounting for finite samples?

The project explores finite-sample confidence bounds and multiple-testing considerations, including:

- Clopper-Pearson confidence intervals
- Multiple-testing corrections
- Bonferroni correction
- Fixed-sequence testing
- Learn-then-Test style procedures

### 6. Does the methodology generalize beyond synthetic corruptions?

CIFAR-10.1 is used as an additional distribution-shift evaluation to determine whether the observations extend beyond the artificial corruptions in CIFAR-10-C.

---

## Experimental Progression

The notebooks are organized as a sequence of experimental phases.

### Phase 1–6: Baseline and Distribution Shift

These experiments establish:

1. Model training and evaluation.
2. Confidence calibration.
3. Threshold selection.
4. Clean-distribution selective prediction.
5. Evaluation under CIFAR-10-C distribution shifts.
6. Comparison across corruption types and severities.

### Phase 7: Initial Statistical Risk Control

The project introduces statistical procedures intended to guarantee the target risk under the shifted distribution.

Initial experiments expose issues related to finite samples and threshold selection.

### Phase 8: Multiple-Testing Correction

Multiple candidate thresholds create a statistical selection problem.

The experiments investigate corrections such as Bonferroni adjustment to preserve validity.

A major observation is that overly conservative corrections can substantially reduce useful coverage.

### Phase 9: Fixed-Sequence Testing

Fixed-sequence testing is explored as an alternative way to control the statistical procedure while retaining more coverage.

The experiments also reveal important implementation and threshold-ordering considerations.

### Phase 10: Natural Distribution Shift

The methodology is evaluated on **CIFAR-10.1** to test whether the observed behavior is limited to synthetic corruptions.

### Phase 11: Refined Threshold Selection

The threshold-selection procedure is refined to address non-monotonic behavior and improve the relationship between:

- statistical validity,
- label budget,
- threshold selection,
- and achievable coverage.

---

## Statistical Framework

The core objective is to select a confidence threshold \(t\) such that the risk of accepted predictions satisfies:

\[
R(t) \leq \alpha
\]

where:

- \(R(t)\) is the selective risk at threshold \(t\),
- \(\alpha\) is the desired risk level.

For a 5% target:

\[
\alpha = 0.05
\]

Because the true deployment risk is unknown, the procedure uses labelled samples from the shifted distribution to construct a finite-sample upper confidence bound.

A threshold is considered acceptable only when the statistical upper bound is consistent with the desired risk level.

This creates a trade-off:

```text
More conservative threshold
        ↓
Lower risk
        ↓
Lower coverage

Less conservative threshold
        ↓
Higher coverage
        ↓
Greater risk / weaker guarantee
```

The research therefore seeks the largest useful coverage that can still be supported by a statistically valid risk guarantee.

---

## Main Research Insight

The central finding investigated by this project is that **confidence calibration on the original distribution should not automatically be interpreted as a deployment-time risk guarantee under distribution shift**.

A classifier can remain confident on shifted inputs while making substantially more errors.

Therefore:

```text
High confidence
        ≠
Guaranteed low deployment risk
```

The project further investigates whether labelled samples from the deployment distribution can restore a statistically defensible guarantee.

---

## Repository Structure

The repository contains a sequence of notebooks corresponding to the experimental phases.

A typical workflow is:

```text
Training
   ↓
Calibration
   ↓
Clean threshold selection
   ↓
Distribution-shift evaluation
   ↓
Label-free methods
   ↓
Label-budget experiments
   ↓
Finite-sample risk guarantees
   ↓
Statistical corrections
   ↓
Natural distribution-shift evaluation
   ↓
Refined threshold-selection procedure
```

The exact notebook names and ordering should be preserved when reproducing the experiments.

---

## Reproducibility

### Requirements

The experiments are implemented primarily in Python/Jupyter notebooks.

A typical environment requires packages such as:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install torch torchvision
jupyter
```

Additional dependencies may be required depending on the individual notebook.

### Running the Experiments

1. Clone or download the repository.
2. Install the required Python dependencies.
3. Start Jupyter:

```bash
jupyter notebook
```

4. Run the notebooks in their intended phase order.
5. Preserve random seeds when reproducing reported experiments.
6. Record the generated metrics, tables, and figures.

---

## Evaluation Metrics

The experiments primarily consider:

### Risk

The fraction of accepted predictions that are incorrect:

\[
\text{Risk}
=
\frac{\text{Incorrect accepted predictions}}
{\text{All accepted predictions}}
\]

### Coverage

The fraction of samples for which the model does not abstain:

\[
\text{Coverage}
=
\frac{\text{Accepted predictions}}
{\text{Total predictions}}
\]

### Target Risk

The maximum acceptable selective risk, typically:

\[
\alpha = 0.05
\]

### Label Budget

The number of labelled examples available from the shifted deployment distribution.

The experiments study how this budget affects the ability to obtain valid risk guarantees and useful coverage.

---

## Comparison With an Oracle

Where applicable, the experiments compare practical threshold-selection methods against an **oracle threshold**.

The oracle has access to the true labels of the shifted distribution and therefore represents an upper reference point for what could be achieved with complete deployment-distribution information.

The oracle is not intended to be a deployable method. It provides a benchmark for evaluating the cost of operating without full deployment labels.

---

## Important Methodological Considerations

The project explicitly considers several statistical issues that can otherwise lead to misleading risk guarantees:

- Distribution shift between calibration and deployment data.
- Finite sample uncertainty.
- Multiple candidate thresholds.
- Multiple hypothesis testing.
- Selection bias.
- Cases with very low or zero coverage.
- Threshold ordering.
- The relationship between label budget and statistical power.
- The distinction between empirical risk and a statistically valid risk guarantee.

These considerations are important because simply finding an empirical error rate below 5% does not establish that the true deployment risk is below 5%.

---

## Expected Research Contribution

The project aims to provide an empirical and statistical analysis of:

1. The failure of clean-distribution confidence thresholds under distribution shift.
2. The relationship between confidence, selective risk, and coverage after shift.
3. The limitations of label-free threshold-selection methods.
4. The usefulness of labelled deployment samples for recovering risk guarantees.
5. The effect of label budget on achievable coverage.
6. The role of finite-sample statistical testing in reliable selective prediction.
7. The generality of these observations across model architectures and distribution shifts.

---

## Caveat About Interpretation

The experimental results should be interpreted according to the exact numerical outputs produced by the notebooks.

In particular, claims about:

- the exact amount of required labelled data,
- exact risk values,
- exact coverage,
- statistical significance,
- superiority of one method over another,
- and generalization to other datasets

should only be made after the corresponding experiments have been executed and their outputs verified.

The repository is therefore intended to make the experimental methodology and statistical reasoning reproducible rather than to imply conclusions beyond the reported evidence.

---

## Research Question in One Sentence

> **How can we obtain statistically valid selective-risk guarantees for neural classifiers when the deployment distribution differs from the distribution used for confidence calibration?**

---

## License

Add the appropriate license for the project before public release.

## Citation

If this work is published, replace the placeholder below with the final citation:

```bibtex
@article{selective_prediction_distribution_shift,
  title   = {Selective Prediction Under Distribution Shift},
  author  = {Shayam Ahmad},
  year    = {2026},
  note    = {Research manuscript}
}
```
