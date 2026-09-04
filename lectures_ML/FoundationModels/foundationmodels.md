# Foundation Models for Weather and Climate

```{admonition} Unit under development
:class: note
The lecture notes and tutorials for this unit are still being written. The
outline below describes the material this unit will cover.
```

Over the last few years, large machine-learned models trained on reanalysis data
have gone from research curiosities to systems that compete with (and on some measures beat) traditional numerical weather prediction, at a tiny fraction of the
inference cost.

This unit covers what these models are, what changed to make them work, and how to
evaluate them honestly.

## Planned topics

- What "foundation model" means in this context, and how it differs from usage
  elsewhere in machine learning
- GraphCast, Pangu-Weather, Aurora, ClimaX, AIFS: architectures and what
  distinguishes them
- Training on reanalysis: what ERA5 is, and what it means that these models learn
  from an analysis product rather than from observations
- How these models are scored, and the ongoing argument about whether standard
  scores flatter them
- Known weaknesses: extremes, physical consistency, and long rollouts
- Using a pretrained model rather than training one, which is what most
  practitioners will actually do
