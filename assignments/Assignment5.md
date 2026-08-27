# Assignment 5: Air sensor bias correction model validation

```{admonition} Not yet released
:class: note
This assignment has not been posted yet. It is due **Monday, October 19** and covers the
material from **Week 5 — Model Evaluation and Validation**. The outline below is provisional.
```
## What this assignment will cover

Low-cost air quality sensors are cheap enough to deploy densely, but they drift and
respond to humidity and temperature, so they are usually corrected against a reference monitor.
That correction is a machine learning model, and validating it properly is harder than it looks.

Starting from the air quality data used in Assignment 4, you will:

- Build a bias correction model that maps low-cost sensor readings to reference values
- Establish the baselines it has to beat — the raw uncorrected reading, and a simple linear
  calibration
- Design a validation scheme that reflects how the model will actually be used: correcting a
  sensor at a **new location**, or over a **later time period**, is not the same test as
  correcting held-out rows from the same deployment
- Compare the score under a naive random split against the score under your chosen scheme, and
  account for the difference
- Audit your pipeline for leakage, including any scaling or imputation fitted before splitting

## Prerequisites

Assignment 4, and the *Model Evaluation and Validation* unit.

## Notes

The gap between the naive and the honest estimate is the point of this assignment.
Reporting only the flattering number will cost you marks, and a correction model that fails to
generalize to a new sensor is a legitimate and useful finding.

## Submitting

Push your completed notebook to your `ml4climate2026` repository, as described in
Assignment 1.
