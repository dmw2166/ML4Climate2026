# Assignment 3: Re-validating a Model Honestly

```{admonition} Not yet released
:class: note
This assignment has not been posted yet. It will be released when the
**Model Evaluation and Validation** unit is covered in class. The outline below is provisional and
describes what the assignment is expected to ask you to do.
```

## What this assignment will cover

You will revisit the random forest you trained in **Assignment 2** and ask a
harder question about it: how much of that performance was real?

You will:

- Establish appropriate baselines for the target variable, and report your model
  against them rather than in isolation
- Re-split the data using a scheme that respects its structure, rather than a
  random split over rows
- Compare the original reported score with the score under honest validation, and
  quantify the difference
- Check whether any preprocessing step saw data it should not have
- Write a short assessment of whether the model's original result would survive
  peer review, and what you would need to change to make it defensible

## Prerequisites

Assignment 2, and the *Model Evaluation and Validation* unit.

## Notes

No new dataset is involved. The point of this assignment is that re-analyzing
a model you already built is usually more instructive than building another one.

## Submitting

Push your completed notebook to your `ml4climate2026` repository, as described in
Assignment 1, Part 6.
