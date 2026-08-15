# Assignment 7: Land Cover Classification from Satellite Imagery

```{admonition} Not yet released
:class: note
This assignment has not been posted yet. It will be released when the
**Remote Sensing and Geospatial ML** unit is covered in class. The outline below is provisional and
describes what the assignment is expected to ask you to do.
```

## What this assignment will cover

You will build a land cover classifier on a real satellite scene, rather than the
synthetic one used in the lecture notebook.

You will:

- Obtain a Sentinel-2 or Landsat scene and prepare the bands you need
- Handle clouds and missing data honestly, rather than quietly dropping pixels
- Compute spectral indices and use them alongside the raw bands
- Train a classifier and validate it with **spatial blocks**, not a random pixel split
- Report both the random-split and blocked accuracy, and explain the difference
- Identify which classes your model confuses, and say whether the fix is a better
  model or better data

## Prerequisites

Assignments 2 and 3, and the *Remote Sensing and Geospatial ML* unit.

## Notes

The gap between random-split and blocked accuracy is part of what is being
assessed. Reporting only the flattering number will cost you marks.

## Submitting

Push your completed notebook to your `ml4climate2026` repository, as described in
Assignment 1, Part 6.
