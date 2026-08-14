# Model Evaluation and Validation

Once we have trained a model, we need to answer a deceptively simple question: **is it any good?**
Answering it correctly turns out to be harder for climate and environmental data than for most
of the datasets used to teach machine learning, and getting it wrong is one of the most common
sources of published results that do not hold up.

## Why the standard recipe fails here

The usual approach is to hold out a random subset of the data, train on the rest, and report
performance on the held-out portion. This works when samples are **independent and identically
distributed** — when knowing one sample tells you nothing about another.

Environmental data violate this assumption almost everywhere:

- **In space.** Two adjacent grid cells, or two nearby monitoring stations, record nearly the
  same values. They are not independent samples; they are close to duplicates.
- **In time.** Today's temperature, soil moisture, or pollutant concentration is strongly related
  to yesterday's. A time series of daily observations contains far fewer independent pieces of
  information than it has rows.
- **Across the sample.** Satellite retrievals from the same orbit, measurements from the same
  instrument, or model output from the same ensemble member all share structure.

When samples are correlated, a random split places near-copies of the test data into the training
set. The model can score well by effectively looking up answers rather than learning a
relationship, and the reported skill is inflated — sometimes dramatically.

## Baselines

A performance number in isolation means nothing. It only becomes interpretable next to a baseline.
Two are conventional in climate and weather work:

- **Climatology** — predict the long-term average for that location and time of year
- **Persistence** — predict that conditions will stay as they are now

These are often surprisingly hard to beat, and that is the point. A model that explains most of
the variance in a temperature series may still be adding nothing over persistence. Reporting your
model's skill without reporting the baselines is an incomplete result.

## Validation should mirror the application

The most useful principle in this whole topic: **the gap between your training and test data
should reflect the gap between your training data and the situation where the model will
actually be used.**

- Applying a model to a region with no training data? Hold out entire regions.
- Forecasting forward in time? Test only on periods after the training window.
- Deploying to a new station or instrument? Hold out whole stations.
- Projecting a future climate state? Ask whether your model can extrapolate at all.

That last case is specific to this field and deserves emphasis. Much of climate science asks
models to make predictions about conditions that have never been observed — warmer mean states,
higher greenhouse gas concentrations, more extreme events than are in the historical record. Some
model families handle this reasonably; others fail completely and without warning. Knowing which
you are working with matters more than squeezing out another point of validation skill.

## Leakage

**Data leakage** is the general name for information from the test set influencing the trained
model. Spatial and temporal correlation are two routes, but there are others that are easy to
introduce by accident:

- Standardizing or normalizing features using statistics computed over the whole dataset
- Selecting features, or tuning hyperparameters, using the full dataset before splitting
- Imputing missing values using information that spans the split

The defense is to treat every preprocessing step as part of the model. Scikit-learn's `Pipeline`
exists for exactly this reason: it ensures each step is fit only on the training fold.

A closing heuristic worth remembering: **if a result seems too good, it probably is.** In this
field, an unexpectedly high score is more often a sign of leakage than a breakthrough, and the
productive response is to go looking for the mistake.
