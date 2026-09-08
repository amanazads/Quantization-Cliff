# PS-1 scorer validation against human labels

- rows exported: 80
- rows labelled: 80  (unlabelled/undecidable: 0)
- raters: AA

## Scorer vs human

- Cohen's kappa: **0.471** (moderate)
- raw agreement: 77.5% [67.2%, 85.3%]
- precision: 75.0%
- recall: 53.6%

- confusion: TP=15 FP=5 TN=47 FN=13

## Caveats

- Human labels are one or two people's judgement, not ground truth. With a single rater there is no way to separate scorer error from rater error.
- Agreement measured on this subset is assumed to hold on the full suite. That assumption is only as good as the subset's stratification.
- Scorer error is CONSTANT across precision arms, so it biases absolute violation rates more than it biases the between-precision comparison that PS-5 asks about. It does not licence quoting the absolute rate as a safety rate.
