# Assignment 3: Hurricane intensity classification and wind speed regression from storm records

```{admonition} Not yet released
:class: note
This assignment has not been posted yet. It is due **Monday, October 5** and covers the
material from **Week 3 — Supervised Learning: Classification and Regression**. The outline below is provisional.
```
## What this assignment will cover

You will return to the tropical cyclone records you cleaned in Assignment 2 and model
them, using both halves of this week's material on the same dataset.

You will:

- Fit a **linear regression** to predict maximum sustained wind speed from storm and
  environmental characteristics, and interpret the coefficients physically
- Fit a **logistic regression** for a binary outcome — for example, whether a storm reaches
  major hurricane status — and interpret the decision boundary
- Extend to the full Saffir-Simpson categories with **softmax regression**
- Compare against a **support vector machine**, and describe where the two approaches differ
- Report appropriate metrics for each task, and explain why accuracy alone is a poor summary
  for the category problem, given how unevenly the categories are populated

## Prerequisites

Assignment 2, and the Week 3 lecture material.

## Notes

Interpretation carries as much weight as implementation. A model with a lower score that
you can explain physically is worth more here than a better one you cannot.

## Submitting

Push your completed notebook to your `ml4climate2026` repository, as described in
Assignment 1.
