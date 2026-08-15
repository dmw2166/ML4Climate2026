# Assignment 9: Statistical Downscaling

```{admonition} Not yet released
:class: note
This assignment has not been posted yet. It will be released when the
**Forecasting, Downscaling, and Extremes** unit is covered in class. The outline below is provisional and
describes what the assignment is expected to ask you to do.
```

## What this assignment will cover

Climate models produce output on grids too coarse for most impact and adaptation
questions. **Downscaling** is the problem of getting from a coarse field to a finer
one that is physically plausible and statistically consistent.

You will:

- Train a model that maps a coarse-resolution field to a finer-resolution target
- Compare it against the appropriate baselines, including simple interpolation
- Validate on a held-out *future* period rather than a random subset of years
- Assess whether the downscaled output preserves the statistics that matter —
  extremes and variability, not only the mean
- Discuss whether your approach would remain valid under a climate state warmer
  than anything in the training record

## Prerequisites

Assignments 3 and 5, and the *Forecasting, Downscaling, and Extremes* unit.

## Notes

Beating interpolation is harder than it sounds, and a downscaling model that
matches the mean while destroying the variance is a common and subtle failure.

## Submitting

Push your completed notebook to your `ml4climate2026` repository, as described in
Assignment 1, Part 6.
