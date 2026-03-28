# Experiment Review: Multi-CWE Fine-Tuning on Expanded BigVul Subset

## Run scope

This experiment fine-tuned the classifier on the expanded multi-CWE BigVul subset and evaluated it on:

- `test`: standard held-out test split over seen CWE setting
- `unseen_test`: held-out test split over unseen CWE setting

This review uses only the valid notebook outputs and **ignores analyses that used the wrong split for selection or tuning**, especially threshold-related conclusions that relied on `test` or `unseen_test` when they should not have.

---

## Executive summary

The main result is strong:

- **test F1:** `0.8774`
- **unseen_test F1:** `0.8737`

The gap between seen and unseen CWE performance is very small. The model does **not** collapse on held-out CWE families. Instead, on unseen CWE it becomes **more conservative**:

- precision goes **up**
- recall goes **down**
- F1 stays nearly the same

This suggests the model learned vulnerability-relevant code patterns that transfer beyond the training CWE set.

The remaining weakness is mainly **missed vulnerable examples** on unseen CWE, not false-positive explosion.

---

## Headline metrics

### Test

- Accuracy: `0.9945`
- Precision: `0.9208`
- Recall: `0.8378`
- F1: `0.8774`

### Unseen test

- Accuracy: `0.9861`
- Precision: `0.9555`
- Recall: `0.8048`
- F1: `0.8737`

### Seen vs unseen deltas

- Δ Accuracy: `-0.0084`
- Δ Precision: `+0.0347`
- Δ Recall: `-0.0331`
- Δ F1: `-0.0037`

### Interpretation

The tiny F1 drop combined with higher unseen precision and lower unseen recall indicates:

- strong transfer to unseen CWE
- conservative behavior on unseen classes
- the main generalization gap is **false negatives**, not false positives

---

## Statistical stability

Bootstrap confidence intervals support the conclusion that unseen performance remains close to seen performance.

### Test bootstrap intervals

- F1 CI: approximately `0.8256 – 0.9231`
- Recall CI: approximately `0.7685 – 0.9065`

### Unseen test bootstrap intervals

- F1 CI: approximately `0.8573 – 0.8904`
- Recall CI: approximately `0.7785 – 0.8310`

### Interpretation

Unseen F1 remains in the same range as seen test F1. The clearer shift is in recall rather than in precision or overall discrimination.

---

## Seen vs unseen CWE behavior

### Seen CWE

- Precision: `0.9208`
- Recall: `0.8378`
- F1: `0.8774`

### Unseen CWE

- Precision: `0.9555`
- Recall: `0.8048`
- F1: `0.8737`

### Error-rate interpretation

- Seen false negative rate: about `0.1622`
- Unseen false negative rate: about `0.1952`

The unseen gap is primarily a **missed-positive problem**. The model appears stricter before calling a function vulnerable when the CWE family is unseen.

---

## Error concentration by CWE

The largest false-negative counts appear in broad and semantically diverse vulnerability families.

### Highest false-negative CWE counts

- `CWE-125`: `30`
- `CWE-119`: `28`
- `CWE-20`: `19`
- `CWE-190`: `14`
- `CWE-200`: `14`
- `CWE-787`: `11`
- `CWE-476`: `10`

### Highest false-positive CWE counts

- `CWE-200`: `5`
- `CWE-416`: `5`
- `CWE-119`: `4`
- `CWE-20`: `4`

### Interpretation

The difficult classes are mostly:

- memory-safety families
- input-validation families
- broad semantic families rather than narrow patterns

Even for the hardest CWEs, the dominant issue is still **false negatives**, not runaway false positives.

---

## Error concentration by project

Errors are concentrated in a relatively small set of large and diverse projects.

### Highest false-negative project counts

- `php-src`: `34`
- `openssl`: `28`
- `FFmpeg`: `21`
- `krb5`: `12`
- `openjpeg`: `11`
- `radare2`: `10`

### Highest false-positive project counts

- `openssl`: `5`
- `php-src`: `5`
- `radare2`: `4`
- `FFmpeg`: `3`
- `wolfssl`: `3`
- `ghostscript`: `3`

### Interpretation

The model’s remaining weakness is concentrated in a few high-diversity projects. These projects are the best targets for manual error inspection and future targeted improvements.

---

## Length and context effects

Length stratification is one of the clearest findings in the analysis.

### Token-length behavior

- `0–64` tokens:
  - Recall: `0.2800`
  - F1: `0.3784`

- `64–128` tokens:
  - Recall: `0.8242`
  - F1: `0.8929`

- `128–256` tokens:
  - Recall: `0.9205`
  - F1: `0.9419`

- `256–512` tokens:
  - Recall: `0.7971`
  - F1: `0.8678`

### False negatives by token bucket

- `256–512`: `142` FN

### Interpretation

Two different failure modes appear:

1. **Very short functions** are difficult because they often do not expose enough local context.
2. **Longer functions near the token ceiling** also produce many misses, likely because they are intrinsically harder or partially truncated.

This suggests context remains a major factor even after moving to function-level modeling.

---

## Code-size effects beyond tokens

The same pattern appears with line count and character length.

### Examples

- `char length > 4096`:
  - Recall: `0.4394`
  - F1: `0.6105`

- `line count 128–256`:
  - Recall: `0.5714`
  - F1: `0.7018`

### Interpretation

Performance is strongest in moderate-size functions and weaker at the extremes, especially for the largest code units.

---

## Rare-class behavior

Rare classes remain unstable.

### Observed pattern

- Non-rare CWE F1: about `0.8753`
- Rare CWE F1: about `0.7143`

### Interpretation

This should be interpreted carefully:

- rare classes have small support
- aggregate performance is carried mostly by supported classes
- rare-class metrics are noisy and should not be over-read

The main takeaway is not that rare CWEs are solved poorly in a definitive sense, but that they remain statistically unstable.

---

## Per-CWE interpretation notes

The per-CWE ranking contains some classes with very low support, including cases with:

- no positives
- one positive
- very few samples overall

These rows should not drive high-level conclusions.

The more trustworthy per-CWE signal comes from supported classes such as:

- `CWE-189`
- `CWE-269`
- the larger memory-safety and input-validation families

### Interpretation

Per-CWE analysis supports the overall story:

- some classes are very strong
- the largest remaining challenge is in broad, supported, semantically varied CWEs

---

## Manual sample review

The displayed false-negative and false-positive samples match the quantitative trends.

### False negatives

Common characteristics:

- many have very low predicted scores
- many are long functions
- many appear in the same difficult projects and CWE families highlighted above

This suggests these are often genuine hard misses, not just threshold-borderline cases.

### False positives

There are also some high-confidence false positives, especially in:

- `poppler`
- `openssl`
- `ghostscript`
- `php-src`

This suggests the model has learned some patterns that it strongly associates with vulnerability, sometimes incorrectly.

---

## Important caveat: unseen_val inconsistency

The notebook reports a much weaker `unseen_val` result:

- unseen_val F1: `0.5308`
- Precision: `0.3931`
- Recall: `0.8172`

This differs dramatically from `unseen_test`.

Because some notebook sections used the wrong split improperly, this review does **not** use `unseen_val` for conclusions. However, the discrepancy is too large to ignore operationally.

Possible explanations include:

- split handling bug
- thresholding mismatch
- prediction file mismatch
- materially different unseen validation distribution

### Action

Do not use `unseen_val` in the project narrative until this discrepancy is explained.

---

## Valid conclusions

Using only the valid parts of the notebook, the supported conclusions are:

1. **Generalization to unseen CWE is strong.**  
   The model retains nearly identical F1 on `unseen_test` compared with `test`.

2. **The main unseen gap is lower recall, not lower precision.**  
   The model becomes more conservative on unseen CWE.

3. **Errors are concentrated in a few broad CWE families and large projects.**

4. **Context and function length remain major factors.**  
   Both very short functions and large near-limit functions are harder.

5. **Rare classes remain unstable and should be interpreted with caution.**

---

## Recommended project narrative

A concise summary suitable for reports:

> On held-out unseen CWE categories, the model maintained nearly the same F1 as on seen test data (`0.8737` vs. `0.8774`), with higher precision but lower recall. This suggests robust transfer of vulnerability-relevant code patterns beyond the training CWE set. The remaining gap is driven primarily by conservative predictions on unseen classes rather than a collapse in discrimination.

A slightly fuller version:

> Fine-tuning on the expanded multi-CWE BigVul subset produced strong results on both standard and unseen-CWE evaluation. Performance transferred well to held-out CWE families, with only a minimal F1 drop from `0.8774` to `0.8737`. On unseen CWE, precision increased while recall decreased, indicating a conservative prediction shift rather than a failure to generalize. Remaining errors are concentrated in broad memory-safety and input-validation CWEs, a handful of large projects, and functions at both the very short and near-truncation ends of the length distribution.

---

## What this means for next experiments

The next question is no longer:

- can the model generalize at all to unseen CWE?

The next questions are:

- how can unseen recall be improved?
- how can short-function and long-function failures be reduced?
- how can errors on the large hard CWE families be reduced?

### Suggested priorities

1. Improve recall on unseen CWE without losing too much precision
2. Analyze representation limits on very short functions
3. Analyze truncation/context effects for long functions near the token ceiling
4. Inspect hard projects such as `php-src`, `openssl`, and `FFmpeg`
5. Reconcile the `unseen_val` inconsistency before using it in future selection logic

---

## Open issues

- Verify split handling for `unseen_val`
- Audit threshold-selection code to ensure no future leakage
- Add split-specific per-CWE/per-project outputs if not already available
- Save manual-review FN/FP samples as CSV for cross-run comparison

---

## Bottom line

This experiment is a real positive result.

The model appears to learn vulnerability-relevant code patterns that transfer to unseen CWE families. The main remaining weakness is conservative behavior on unseen classes, visible as lower recall rather than loss of precision.