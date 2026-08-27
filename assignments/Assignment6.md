# Assignment 6: Extracting modes of climate variability with PCA/EOF

```{admonition} Not yet released
:class: note
This assignment has not been posted yet. It is due **Monday, October 26** and covers the
material from **Week 6 — Unsupervised Learning: Dimensionality Reduction**. The outline below is provisional.
```
## What this assignment will cover

Climate fields are high-dimensional and strongly correlated in space, which is exactly
the situation dimensionality reduction was built for. In climate science the principal components
of a spatial field are called **empirical orthogonal functions**, and the leading ones often
correspond to named modes of variability.

You will:

- Apply principal component analysis to a gridded climate field, such as sea surface temperature
- Interpret the leading modes physically — check whether the leading EOF corresponds to a mode
  you can name, and examine its principal component time series against a known index
- Examine the explained variance spectrum and justify how many components to retain
- Reconstruct the field from a truncated set of components and quantify what was lost
- Discuss what PCA assumes — linearity, orthogonality, and that variance means importance — and
  where those assumptions are questionable for geophysical data

## Prerequisites

Assignments 2 and 5, and the Week 6 lecture material.

## Notes

The physical interpretation is the part that matters. Any library will return
eigenvectors; saying what they mean, and checking that claim against an independent index, is
the skill being assessed.

## Submitting

Push your completed notebook to your `ml4climate2026` repository, as described in
Assignment 1.
