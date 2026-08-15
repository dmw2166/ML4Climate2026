# Uncertainty Quantification

```{admonition} Unit under development
:class: note
The lecture notes and tutorials for this unit are still being written. The
outline below describes the material this unit will cover.
```

Every model output is uncertain. In climate science that uncertainty is not a
footnote — it is often the result. A projection without an honest error estimate
cannot support a decision, and a confident wrong answer is worse than an uncertain
right one.

This unit covers how to produce uncertainty estimates from machine learning models,
and how to check whether those estimates mean anything.

## Planned topics

- Sources of uncertainty: aleatoric versus epistemic
- Ensembles as a practical route to uncertainty estimates
- Calibration: whether your 90% intervals actually contain the truth 90% of the time
- Conformal prediction, and distribution-free coverage guarantees
- Quantile regression and predicting distributions rather than point values
- Communicating uncertainty to non-specialist audiences without either
  overstating or burying it
