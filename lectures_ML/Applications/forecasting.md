# Forecasting, Downscaling, and Extremes

```{admonition} Unit under development
:class: note
The lecture notes and tutorials for this unit are still being written. The
outline below describes the material this unit will cover.
```

This unit brings together three applied problems that share a common structure:
each asks a model to produce something finer, further ahead, or rarer than what it
was directly trained on.

They are grouped because they are where the methods from the rest of the course meet
questions that practitioners actually get asked.

## Planned topics

- Subseasonal-to-seasonal prediction, and why the S2S range is hard
- ENSO prediction as a tractable, well-documented case study
- Statistical downscaling: from coarse model output to usable resolution
- Bias correction, and the assumptions it quietly makes
- Detecting and characterizing extreme events
- Why extremes are the hardest case for machine learning: they are rare by
  definition, so the training data contains fewest examples of what matters most
