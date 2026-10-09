# Model Evaluation and Validation

Once we have trained a model, we need to answer a deceptively simple question: **is it any good?**
Answering it correctly turns out to be harder for climate and environmental data than for most
of the datasets used to teach machine learning, and getting it wrong is one of the most common
sources of published results that do not hold up.

This page is the overview. The [next page](validation.ipynb) works through each idea in
code: baselines, why one hold-out score is not enough, k-fold cross-validation, folds that
respect space and time, and hyperparameter tuning. The tutorial then runs the whole recipe
on river flow data.

## Why the standard recipe fails here

The usual approach is to hold out a random subset of the data, train on the rest, and report
performance on the held-out portion. This works when samples are **independent and identically
distributed**: when knowing one sample tells you nothing about another.

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
relationship, and the reported skill is inflated, sometimes dramatically.

## Baselines

A performance number in isolation means nothing. It only becomes interpretable next to a baseline.
Two are conventional in climate and weather work:

- **Climatology**: predict the long-term average for that location and time of year
- **Persistence**: predict that conditions will stay as they are now

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

A fourth case, projecting into a climate state that has never been observed, is not a
validation problem at all: no split can test it, and Week 5 showed that some model families
cannot do it. Know which family you are using before you ask.

## One split is not enough

Even when a random split is the right split, the score it gives depends on which samples
landed in the test set. **K-fold cross-validation** cuts the data into $k$ folds, holds out
each in turn, and reports the mean and spread of the $k$ scores. It uses every sample for
testing once and most of the data for training every time. On correlated data the folds are
built with `GroupKFold` or `TimeSeriesSplit` instead of a shuffle, but the mechanics are the
same. The next page shows the loop, the scikit-learn shortcuts, and the choices of $k$.

## Hyperparameters are chosen on data the score never sees

Settings such as a tree's depth or a neighbor count are not learned by `fit`. They are chosen
by trying values and keeping the best, and whatever data make that choice can no longer give
an unbiased score. Either hold back a validation set for the choice and a test set for the
score, or run the search by cross-validation on the training data with `GridSearchCV` and
test once at the end. The splitter inside the search has to be the same one the validation
uses; a search with shuffled folds on a time series picks the model that is best at copying
its neighbors.

## Leakage

**Data leakage** is the general name for information from the test set influencing the trained
model. Spatial and temporal correlation are two routes, but there are others that are easy to
introduce by accident:

- Standardizing or normalizing features using statistics computed over the whole dataset
- Selecting features, or tuning hyperparameters, on data that includes the test set
- Imputing missing values using information that spans the split

The defense is to treat every preprocessing step as part of the model. Scikit-learn's `Pipeline`
exists for exactly this reason: it ensures each step is fit only on the training fold.

A closing heuristic worth remembering: **if a result seems too good, it probably is.** In this
field, an unexpectedly high score is more often a sign of leakage than a breakthrough, and the
productive response is to go looking for the mistake.
