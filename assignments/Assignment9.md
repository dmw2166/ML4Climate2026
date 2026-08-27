# Assignment 9: Cloud type classification from satellite imagery with CNNs

```{admonition} Not yet released
:class: note
This assignment has not been posted yet. It is due **Monday, November 23** and covers the
material from **Week 10 — Deep Learning: Convolutional and Recurrent Networks**. The outline below is provisional.
```
## What this assignment will cover

Cloud classification is a natural convolutional problem: cloud types are defined by
texture and spatial organization rather than by the brightness of any single pixel, which is
precisely the structure a CNN is built to exploit.

You will:

- Train a convolutional network to classify cloud types from satellite imagery
- Examine what the learned filters respond to, and whether those features are physically sensible
- Apply data augmentation, and measure whether it helps
- Try transfer learning from a pretrained network, and compare against training from scratch
- Validate in a way that reflects the spatial structure of the data — neighbouring image tiles
  are not independent samples
- Report which cloud types are confused with one another, and say whether the fix is a better
  model or better data

## Prerequisites

Assignment 8, and the Week 10 lecture material.

## Notes

Confusion between visually similar cloud types is expected and is not in itself a
failure. Diagnosing *why* particular pairs are confused is worth more than a higher overall
accuracy.

## Submitting

Push your completed notebook to your `ml4climate2026` repository, as described in
Assignment 1.
