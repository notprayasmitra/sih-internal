# CTU-13 Evaluation Role

CTU-13 is not used for supervised stage-accuracy validation in the current
experiment. Its flow labels are valid for binary risk targets, but valid labels
are sparse relative to the dominant background traffic. With the configured
0.8 window label-coverage threshold, CTU-13 produces no stage-supervised
windows, so stage accuracy on a CTU-only validation split is undefined.

CTU-13 is used instead for two evaluation purposes:

- Binary malicious-risk and compromise-risk benchmarking against the logistic
  history-window baseline and the world model risk heads.
- Cross-dataset state-generalization reporting, including aggregate state RMSE
  and per-feature state error compared with the CIC-IDS2018 training domain.
  The report separates features observed in the CIC training trajectories from
  features unavailable in CIC and therefore unseen by the fitted normalizer.

This keeps stage validation restricted to trajectories with adequate window-level
stage coverage while treating CTU-13's distribution shift as an explicit result
rather than hiding it in an invalid stage metric.

## Risk benchmark interpretation

On the current CTU-13 malicious-risk run, the world model has very low recall
(`0.085`) and low false-positive rate (`0.099`). The logistic baseline has very
high recall (`0.96`) and very high false-positive rate (`0.96`). Neither model
is well-calibrated on this cross-dataset task yet. F1 alone is not a sufficient
summary; recall and false-positive rate must be reported together in submission
material.
