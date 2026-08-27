# Assignment 10: Generative Models

```{admonition} Not yet released
:class: note
This assignment has not been posted yet. It is due **Monday, December 1** and covers the
material from **Week 11 — Deep Learning: Generative Models**. The outline below is provisional.
```

## What this assignment will cover

Generative models learn a distribution rather than a mapping, which makes them
useful for problems where the goal is plausible samples rather than a single
prediction.

You will:

- Train an autoencoder and inspect what its latent space has organized
- Train a variational autoencoder and sample from it
- Compare the samples against the training distribution — do they reproduce its
  statistics, or only its mean behavior?
- Describe how a GAN and a diffusion model differ in how they are trained, and what
  that implies for stability and sample quality
- Discuss one climate application where generating plausible samples is more useful
  than predicting a single expected value

## Prerequisites

Assignments 8 and 9, and the Week 11 lecture material.

## Notes

Evaluating a generative model is genuinely harder than evaluating a predictive
one, since there is no single right answer to score against. Part of this assignment
is confronting that.

## Submitting

Push your completed notebook to your `ml4climate2026` repository, as described in
Assignment 1.
