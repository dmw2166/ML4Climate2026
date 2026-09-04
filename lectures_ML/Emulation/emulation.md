# Climate Model Emulation

Running a general circulation model is expensive. A single high-resolution simulation can consume
substantial supercomputer time, which puts a hard limit on how many scenarios, parameter settings,
or ensemble members anyone can afford to explore.

**Emulation** addresses this by training a cheap statistical model on a limited set of expensive
simulations, then using that surrogate to predict what the full model would have produced
elsewhere. A good emulator runs in milliseconds instead of core-months, which turns questions that
were previously unaskable into routine ones.

## What an emulator is, and what it is not

An emulator is a model **of a model**. Its training data is simulation output, and its target is
what the simulator would have said, not what the atmosphere would have done. This distinction is
easy to state and easy to forget, and it has two consequences.

First, an emulator **inherits every bias of the simulator it was trained on**. If the GCM runs too
warm over the Southern Ocean, so will the emulator, faithfully. Emulation is not a route to a
better climate model; it is a route to a faster one.

Second, the thing being approximated is deterministic in a way that most machine learning targets
are not. Given identical inputs, a GCM produces identical output (leaving aside internal
variability across ensemble members). So the irreducible noise floor is much lower than in
observational work, and a well-designed emulator can get very close to the simulator. When it does
not, the gap is nearly all epistemic. See [Uncertainty Quantification](../Uncertainty/uncertainty.md).

An emulator is also **not** a general-purpose predictor. It is valid only within the region of
input space the training runs covered, which is the subject of the last section here.

## Two things people mean by "emulator"

The word covers two related but distinct problems, and it is worth being explicit about which one
you are solving, because the design choices differ.

**Parameter emulation.** The inputs are the simulator's own tunable parameters (cloud, convection and microphysics settings) and the output is the climate the model produces. This is
what you build in the [ANN tutorial](../DeepLearning/ann_tutorial.ipynb) in Week 9, using
perturbed parameter ensembles. The purpose is usually **model tuning or calibration**: search the
parameter space cheaply to find settings that best match observations, which is impossible to do
by brute force when each evaluation costs a supercomputer allocation.

**Scenario emulation.** The inputs are forcings (emissions or concentrations of CO$_2$, methane, aerosols) and the output is the climate response. The purpose is **scenario exploration**: the
IPCC process needs many more emissions pathways than anyone can afford to simulate directly, and
an emulator can fill in the ones that were never run.

`ClimateBench` is the standard public benchmark for the second kind. Its emulators take global
annual forcing time series and predict the resulting **spatial field** of surface temperature,
trained on a handful of scenarios and tested on a scenario held out entirely.

## Experimental design: choosing the training runs

Because each training example costs a simulation, the choice of which runs to perform is a
first-class design problem, and it matters more than the choice of statistical model.

The goal is to cover the input space as evenly as possible with as few runs as you can afford.
Randomly sampling parameters wastes runs on clusters and leaves gaps. **Latin hypercube sampling**
and related space-filling designs are the standard answer: they guarantee that each parameter is
sampled across its full range while avoiding the clumping of pure random draws.

Two practical points follow:

- **Design for the space you will predict in.** An emulator interpolates well and extrapolates
  badly, so the training runs must bracket the region of interest on every axis. If you intend to
  ask about high climate sensitivity, the ensemble has to contain high-sensitivity runs.
- **Sample dimensions you will actually vary.** Parameter counts grow quickly, and the number of
  runs needed grows with them. Screening experiments that identify which parameters matter are
  usually worth the cost before committing to a large ensemble.

## Emulating fields, not just global means

Predicting global mean temperature from forcings is a scalar regression and not very hard. Most
of the scientific value is in **spatial** output (where it warms, where precipitation changes, where the extremes shift) and that is a much larger output space. A single 96×144 field is
nearly fourteen thousand numbers, predicted from a handful of inputs.

Three approaches, in rough order of sophistication:

- **Independent per-cell models.** Fit a separate regression for every grid cell. Simple and
  embarrassingly parallel, but it ignores spatial structure entirely and produces noisy,
  incoherent fields. The ClimateBench random forest baseline works essentially this way, by
  stacking latitude and longitude into a multi-output target.
- **Reduce, then emulate.** Compress the field with PCA (the technique from
  [Week 6](../UnsupervisedML/PCA.ipynb)), emulate the handful of principal component
  coefficients, then project back. This exploits the fact that climate fields have far fewer
  degrees of freedom than grid cells, the same insight that made EOF analysis useful in
  Assignment 6.
- **Predict the field directly** with a convolutional decoder, letting the network learn the
  spatial structure. This is what the stronger ClimateBench entries do.

## Uncertainty, and why it is not optional here

An emulator used for tuning or for scenario exploration is feeding a decision, so a point estimate
is rarely enough. Two questions need answering: how much does the emulator disagree with the
simulator, and where in input space is it least sure?

Gaussian processes are historically the default emulator for exactly this reason; they return a
predictive distribution rather than a point, and the uncertainty widens automatically away from
the training runs. That property is worth more here than raw accuracy, because it tells you when
you have wandered outside the design. Neural network emulators are more accurate on large
ensembles but need explicit machinery (ensembles, quantile losses, conformal wrappers) to say
anything about their own confidence.

## Where emulators fail

**Beyond the training runs.** This is the failure that matters, and it is structural rather than
fixable. An emulator has seen the simulator's behaviour only in the region the ensemble covered.
Asked about a forcing pathway or a parameter combination outside that region, it will still return
a confident-looking number, produced by whatever its functional form happens to do out there. The
ClimateBench design confronts this honestly by testing on a **held-out scenario**: train on
`historical`, `ssp126`, `ssp370`, `ssp585` and the single-forcing runs, then predict `ssp245`,
which the emulator has never seen. That is a real generalization test, and it is much harder than
a random split over years.

This connects directly to the point made in Assignment 1: extrapolation is a different problem
from interpolation, and data-driven methods are far weaker at it. For emulation the stakes are
specific: the scenarios policymakers most want costed are often the ones furthest from what has
been simulated.

**Novel physics.** An emulator learns a statistical relationship between inputs and outputs. If
the simulator's behaviour changes qualitatively somewhere (a circulation regime shift, an ice sheet collapse, a tipping point) nothing in the training data anticipates it, and the emulator
will smoothly interpolate straight through the discontinuity.

**Silent staleness.** Emulators are tied to a specific model version. Update the simulator's
parameterizations and the emulator is quietly describing a model that no longer exists, with no
error to signal it.

The honest summary: an emulator is an interpolator over a designed ensemble. Used inside its
design it is enormously powerful and can replace thousands of core-hours. Used outside it, it
fails in the most dangerous way available, confidently, and without complaint.
