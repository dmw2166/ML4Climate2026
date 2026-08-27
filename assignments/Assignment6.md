# Assignment 6: Dimensionality Reduction

```{admonition} Not yet released
:class: note
This assignment has not been posted yet. It is due **Monday, October 20** and covers the
material from **Week 6 — Unsupervised Learning: Dimensionality Reduction**. The outline below is provisional.
```

## What this assignment will cover

Climate data is high-dimensional and highly correlated, which is precisely the
situation dimensionality reduction is designed for.

You will:

- Apply principal component analysis to a spatial climate field
- Interpret the leading modes physically — in climate science these are the
  empirical orthogonal functions, and the leading ones often correspond to known
  modes of variability
- Examine the explained variance and justify how many components to retain
- Reconstruct the field from a truncated set of components and quantify what was lost
- Discuss what PCA assumes, and where those assumptions are questionable for
  geophysical data

## Prerequisites

Assignments 2 and 5, and the Week 6 lecture material.

## Notes

The physical interpretation of the leading modes is the part that matters. Any
library can return eigenvectors; saying what they mean is the skill.

## Submitting

Push your completed notebook to your `ml4climate2026` repository, as described in
Assignment 1.
