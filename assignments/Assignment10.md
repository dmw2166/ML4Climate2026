# Assignment 10: Anomaly detection in air quality sensor data with autoencoders

```{admonition} Not yet released
:class: note
This assignment has not been posted yet. It is due **Monday, November 30** and covers the
material from **Week 11 — Deep Learning: Generative Models**. The outline below is provisional.
```
## What this assignment will cover

An autoencoder trained on normal data learns to reconstruct it well. Feed it something
unusual and the reconstruction degrades — which turns a generative model into an anomaly
detector, without ever needing labelled examples of what "anomalous" looks like.

This is genuinely useful for sensor networks, where faults, drift, and unusual pollution events
all need catching and nobody has labelled them in advance.

You will:

- Train an autoencoder on a period of normal air quality sensor data
- Use reconstruction error to score how anomalous each observation is
- Choose an operating threshold, and justify it in terms of the cost of a false alarm versus a
  missed event
- Compare against simple statistical baselines, such as a rolling z-score
- Distinguish, where you can, between sensor faults and genuine pollution episodes — both raise
  the reconstruction error, and only one is a data quality problem
- Explain how a **variational** autoencoder differs, and what its latent space buys you

## Prerequisites

Assignments 8 and 9, and the Week 11 lecture material.

## Notes

Evaluating an unsupervised detector is harder than evaluating a classifier, because you
have no labels to score against. Confronting that honestly — and saying how you would validate
this if it were deployed — is part of the assignment.

## Submitting

Push your completed notebook to your `ml4climate2026` repository, as described in
Assignment 1.
