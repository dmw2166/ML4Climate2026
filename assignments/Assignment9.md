# Assignment 9: Convolutional and Recurrent Networks

```{admonition} Not yet released
:class: note
This assignment has not been posted yet. It is due **Monday, November 24** and covers the
material from **Week 10 — Deep Learning: CNNs and RNNs**. The outline below is provisional.
```

## What this assignment will cover

This assignment covers both architectures from Week 10, applied to problems that
suit their respective inductive biases.

You will:

- Train a convolutional network on gridded or image-like environmental data
- Examine what the learned filters respond to
- Train a recurrent network (LSTM or GRU) on a time series forecasting problem
- Compare against the persistence baseline at several lead times
- Explain why a CNN suits spatial data and an RNN suits sequential data, in terms
  of the structure each architecture assumes about its input

## Prerequisites

Assignment 8, and the Week 10 lecture material.

## Notes

Beating persistence at short lead times is harder than it looks. If your model
does not beat it, report that honestly and diagnose why.

## Submitting

Push your completed notebook to your `ml4climate2026` repository, as described in
Assignment 1.
