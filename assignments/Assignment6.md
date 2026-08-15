# Assignment 6: Building a Climate Model Emulator

```{admonition} Not yet released
:class: note
This assignment has not been posted yet. It will be released when the
**Climate Model Emulation** unit is covered in class. The outline below is provisional and
describes what the assignment is expected to ask you to do.
```

## What this assignment will cover

Climate models are expensive to run, which limits how much of parameter space
anyone can explore. An **emulator** is a cheap statistical surrogate trained on a
limited set of model runs, used to predict what the full model would have produced.

You will:

- Train an emulator that maps forcing or parameter settings to a climate response
- Evaluate it on held-out runs, including runs outside the training range
- Quantify where the emulator is trustworthy and where it is not
- Use the emulator to explore a region of parameter space the training runs did not
  cover, and state clearly how much confidence that exploration deserves

## Prerequisites

Assignments 2, 3 and 5. Regression, neural networks, and validation.

## Notes

Pay attention to the extrapolation material from the Model Evaluation unit. An
emulator asked about conditions outside its training range is exactly the failure
case discussed there.

## Submitting

Push your completed notebook to your `ml4climate2026` repository, as described in
Assignment 1, Part 6.
