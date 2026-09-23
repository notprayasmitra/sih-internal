# Model results

The deployed demonstration model is the probabilistic LSTM checkpoint from epoch 16 of
`world_model_full_combined_lstm`. It was trained on seven complete trajectories and validated
on four held-out complete trajectories. The dataset contains eleven real trajectories from
CIC-IDS2018 and CTU-13 spanning seven attack families.

## Same-family holdout

Best-observed malicious-risk F1 by held-out trajectory:

| Trajectory | Attack family | F1 | Precision | FPR |
|---|---|---:|---:|---:|
| CIC 21-Feb | DDoS | 0.6319 | 1.0000 | 0.0000 |
| CIC 1-Mar | Infiltration | 0.5685 | 0.6414 | 0.1076 |
| CTU scenario 9 | Neris botnet | 0.6982 | 0.7792 | 0.1870 |
| CTU scenario 11 | Rbot ICMP DoS | 0.5274 | 0.4109 | 0.1693 |

The current logistic-regression comparison is recorded in
`runs/world_model_full_combined_lstm/benchmark.json`.

## Stage-taxonomy coverage limitation

The model's stage taxonomy includes four MITRE-aligned categories for which the current
training corpus (CIC-IDS2018, CTU-13 scenarios 1/9/10/11) provides no defensible flow-level
evidence under our conservative label-mapping policy: `RECONNAISSANCE`, `LATERAL_MOVEMENT`,
`COMMAND_AND_CONTROL`, and `EXFILTRATION`. The model cannot learn to predict these stages from
the current training split. This is a known scope boundary caused by dataset coverage, not a
training failure.

We investigated mining additional stage supervision before changing the loss. CTU-13 contains
explicit numbered command-and-control channel labels, but the selected scenarios still produce
no additional stage-supervised windows under the unchanged 10-second aggregation and 0.8 label-
coverage threshold. CIC 23-Feb repeats the same sparse web-attack families as 22-Feb and is not
expected to materially increase malicious modal-stage windows. Sparse attack-labelled flows are
useful for the independent 1% malicious-risk target, but they rarely become a window's modal
stage. Neither source was therefore added as a retraining input.

The CTU adapter's numbered-CC/IRC matching was corrected separately for flow-level metadata and
reporting accuracy. This correction does not change current training data or model behavior at
the 0.8 stage-label coverage threshold.

## Class-weighted stage-loss experiment

Balanced inverse-frequency stage weights were fitted using only the seven training trajectories.
The resulting weights were 0.2720 for `NORMAL`, 4.1388 for `INITIAL_ACCESS`, 12.7315 for
`IMPACT`, and 334.4924 for `OTHER_MALICIOUS`; absent classes received zero weight. The isolated
candidate stopped after epoch 9. It did not reduce the 100% `OTHER_MALICIOUS` misclassification
rate on correctly risk-flagged CIC 1-Mar windows, and its shared representation regressed risk
F1 on infiltration, Neris, and Rbot. It therefore failed the no-regression guard and was not
adopted. The deployed epoch-16 unweighted checkpoint remains unchanged.

## Architecture extension

Latent JEPA was trained on the identical 7/4 split. The larger dataset prevented its earlier
universal zero-recall collapse and improved Neris F1, but it inverted score separation on CTU
scenario 11. The probabilistic LSTM remains the more reliable universal architecture.

## Temporal Transformer comparison

A small autoregressive Temporal Transformer was implemented with the same 51-feature input,
state encoder, availability-mask handling, prediction heads, 6-window history, 7/4 trajectory
split, 100-epoch cap, and patience-8 early stopping. It used a two-layer, four-head
self-attention backbone and preserved multi-step rollout. Training stopped at epoch 18; the
best validation-loss checkpoint was epoch 10, after 294.94 seconds (about 4m55s).

| Attack family | LSTM F1 / FPR | JEPA F1 / FPR | Temporal Transformer F1 / FPR |
|---|---:|---:|---:|
| DDoS | 0.6319 / 0.0000 | 0.6856 / 0.6120 | **0.6358 / 0.0000** |
| Infiltration | 0.5685 / 0.1076 | 0.5970 / 0.2687 | 0.5730 / 0.1663 |
| Neris botnet | **0.6982 / 0.1870** | 0.8315 / not reported | 0.5983 / 0.8532 |
| Rbot ICMP DoS | **0.5274 / 0.1693** | 0.2428 / 1.0000 | 0.5122 / 0.1114 |

## Temporal Convolutional Network comparison

A causal, dilated residual Temporal Convolutional Network (TCN) was implemented as the fourth
architecture. It reused the exact 51-feature state encoder, availability masks, prediction
heads, six-window history, autoregressive rollout contract, 7/4 trajectory split, 100-epoch cap,
and patience-8 early stopping. The two residual convolution blocks used causal kernel-3
convolutions with dilation 1 and 2 (226,161 parameters). Training stopped naturally at epoch 12;
epoch 4 was selected because it had the clear minimum validation loss (0.2094, versus 0.3293 at
epoch 5 and 4.2932 by epoch 9), after about 138 seconds.

The complete best-observed threshold sweep selected these points:

| Trajectory | Best threshold | Precision | Recall | F1 | FPR | Stage accuracy |
|---|---:|---:|---:|---:|---:|---:|
| CIC 21-Feb DDoS | 0.8 | 1.0000 | 0.4661 | 0.6358 | 0.0000 | 0.6757 |
| CIC 1-Mar infiltration | 0.1 | 0.6614 | 0.4737 | 0.5520 | 0.0914 | 0.7686 |
| CTU scenario 9 Neris | 0.1 | 0.7813 | 0.7062 | 0.7419 | 0.2062 | unavailable |
| CTU scenario 11 Rbot | 0.5 | 0.6596 | 0.4306 | 0.5210 | 0.0356 | unavailable |

The TCN was the closest competitor to the LSTM: it reached near-parity on DDoS, improved Neris
F1, and materially reduced Rbot FPR (0.036 versus 0.169). It nevertheless missed the strict
adoption guard by a small margin on infiltration F1 (0.5520 versus 0.5685) and Neris FPR
(0.2062 versus 0.1870). Epoch-level model checkpoints were not saved for the full run, so a
slightly different epoch selection was not exhaustively ruled out; the clear validation-loss
minimum makes that unlikely to change the conclusion.

## Four-way architecture comparison

| Attack family | LSTM F1 / FPR | JEPA F1 / FPR | Temporal Transformer F1 / FPR | TCN F1 / FPR |
|---|---:|---:|---:|---:|
| DDoS | **0.6319 / 0.0000** | 0.6856 / 0.6120 | 0.6358 / 0.0000 | 0.6358 / 0.0000 |
| Infiltration | 0.5685 / 0.1076 | **0.5970 / 0.2687** | 0.5730 / 0.1663 | 0.5520 / **0.0914** |
| Neris botnet | 0.6982 / 0.1870 | **0.8315 / not reported** | 0.5983 / 0.8532 | 0.7419 / 0.2062 |
| Rbot ICMP DoS | **0.5274 / 0.1693** | 0.2428 / 1.0000 | 0.5122 / 0.1114 | 0.5210 / **0.0356** |

Four architectures were therefore tested: probabilistic LSTM, latent JEPA, Temporal Transformer,
and TCN. JEPA improved selected families but inverted or destabilized others; the Transformer
underperformed materially on CTU-13; and the TCN was close but failed the all-family,
F1/precision/FPR no-regression guard. The probabilistic LSTM remains deployed as the most
consistent performer across all four held-out families. Architecture experimentation is closed;
remaining work is documentation and presentation deliverables.

Stage accuracy was 0.7526 on CIC 21-Feb and 0.7686 on CIC 1-Mar; CTU-13 stage accuracy is
undefined because those trajectories have no valid stage-supervised windows. The Transformer was
comparable to the LSTM on DDoS and infiltration but underperformed materially on both CTU-13
families (Neris FPR 0.85 versus the LSTM's 0.19), consistent with Transformers requiring more
data than was available here to outperform recurrent models on short sequences. Attention
extraction was skipped because the model was not competitive on two of the four families. The
probabilistic LSTM remains the sole deployed model as the most consistent performer across all
four held-out attack families.

## Unseen-family holdout

A second LSTM was trained with both DDoS trajectories removed and validated only on those two
days. It detected meaningful unseen DDoS signal, especially on 21-Feb, but suffered substantial
false positives and poor calibration on 20-Feb. This supports sensitivity to unseen malicious
patterns, not deployment-ready zero-shot classification.

## Calibration and ensembles

Train-only Platt calibration improved CTU scenario 11 from F1 0.5274 to 0.5561 while improving
precision and FPR. It did not harmonize thresholds across all families and was rejected for
infiltration and Neris where F1 regressed. LSTM+JEPA average and max ensembles were also rejected:
each violated the strict no-regression guard on at least one protected family. See
`runs/world_model_full_combined_lstm/calibration_ensemble_analysis.json` for the complete sweeps.
