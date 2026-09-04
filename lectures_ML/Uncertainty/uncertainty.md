# Uncertainty Quantification

Every model output is uncertain. In climate science that uncertainty is not a footnote; it is
often the result. A projection without an honest error estimate cannot support a decision, and a
confident wrong answer is worse than an uncertain right one.

The previous unit asked whether a model is any good. This one asks a harder question: **how wrong
might this particular prediction be, and can I trust that estimate?**

## Two kinds of uncertainty

The distinction that organizes everything else is between uncertainty that comes from the world
and uncertainty that comes from our ignorance of it.

**Aleatoric uncertainty** is irreducible noise in the system. Measurement error, turbulence,
genuinely stochastic processes. If you had infinite training data, this would not go away. A
low-cost air quality sensor has a noise floor; no calibration model, however good, can predict
below it.

**Epistemic uncertainty** is uncertainty about the model itself, arising because we have finite
data. It is largest where the training data is sparse or absent, and (crucially) **it is
reducible**: collect more data in that region and it shrinks.

The practical reason to separate them is that they call for different responses. High aleatoric
uncertainty means you have reached the limit of what is predictable and should report that
honestly. High epistemic uncertainty means you should go and collect more data, or restrict the
model's domain of application.

They also behave differently under extrapolation, which is where this matters most for us. In
Assignment 1 you fit a straight line to the Mauna Loa CO$_2$ record and extrapolate it forward;
the error grows steadily the further you go. That growth is epistemic: the model has left the
region the data constrained. A model that reports *constant* uncertainty when extrapolating is
not reporting uncertainty at all.

## Models that predict a distribution

Most of the models in this course emit a single number. Some can emit a distribution instead,
and those are worth knowing about.

**Gaussian processes** are the cleanest example. Rather than fitting one function, a GP maintains
a distribution over functions consistent with the data, and returns a mean and a standard
deviation at every input. The standard deviation is small near observations and widens where
data is sparse, epistemic uncertainty, made visible.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

gp = GaussianProcessRegressor(kernel=RBF(length_scale=10) + WhiteKernel())
gp.fit(X_train, y_train)

mean, sd = gp.predict(X_new, return_std=True)     # a distribution, not a point
```

The `WhiteKernel` term is doing something specific and worth noticing: it estimates the
*aleatoric* noise separately from the *epistemic* spread captured by the RBF term. Fit a GP to
the CO$_2$ record and extrapolate, and the uncertainty band fans out visibly beyond the last
observation: the model telling you, correctly, that it does not know.

GPs scale poorly (roughly cubically in the number of training points), so they are used for
small datasets, expensive-simulation emulation, and cases where calibrated uncertainty matters
more than raw predictive power.

**Quantile regression** is a cheaper alternative. Instead of predicting the conditional mean,
fit several models predicting the 10th, 50th and 90th percentiles of the target, each with an
asymmetric loss. The spread between them is a predictive interval, and it can vary with the
inputs, wide where the process is noisy, narrow where it is not.

This is a natural fit for the streamflow problem in Assignment 8. There, a network trained on
squared error systematically **under-predicts floods**, because the loss averages over thousands
of ordinary days and the few extreme ones barely register. A quantile model asked for the 95th
percentile answers a different and more useful question for a flood warning: not "what will the
flow be?" but "how high could it plausibly get?"

## Ensembles

The most practical route to uncertainty for the models we actually use is to train several and
look at the spread.

- **Different random seeds**: retrain the same architecture with different initializations.
  Captures uncertainty from the fitting procedure.
- **Different data subsets**: bootstrap the training data and refit. This is what a random
  forest already does internally, which is why the spread across its trees can be read as an
  uncertainty estimate.
- **Different model families**: a forest, a network and a linear model. Disagreement between
  them is informative in a way that disagreement between seeds is not.

Ensembles are the standard approach in weather and climate for a reason: they need no special
model, and they produce a distribution directly. The caveat is that **an ensemble only captures
the uncertainty it was constructed to capture.** Ten networks with different seeds, trained on
the same data with the same architecture, will agree closely with each other and can be
confidently wrong together. Their agreement is not evidence of correctness.

## Calibration

Here is the question that matters, and the one most often skipped: **when the model says 90%, is
it right 90% of the time?**

An uncertainty estimate is only useful if it is *calibrated*. A model whose 90% intervals contain
the truth 50% of the time is not conservative or approximate; it is wrong, in a way that will
propagate silently into every decision made from it.

Checking calibration is straightforward for a regression: compute the fraction of test points
falling inside each nominal interval and compare against the nominal level.

```python
inside = np.abs(y_test - mean) < 1.645 * sd     # nominal 90% interval
print(f"nominal 90%, actual coverage {100 * inside.mean():.1f}%")
```

For classifiers the analogue is a **reliability diagram**: bin the predicted probabilities, and
plot the observed frequency in each bin against the predicted probability. A calibrated model
lies on the diagonal.

You have already met this distinction. In Assignment 3, logistic regression and an SVM both
classify major hurricanes about equally well, but only logistic regression emits a *calibrated
probability*; it is fitted by maximizing the likelihood of the observed labels, which is a
proper scoring rule. The SVM's decision function ranks storms correctly but its units are
arbitrary. For a forecaster deciding whether to order an evacuation, where the costs are wildly
asymmetric and the action threshold is nowhere near 50%, that difference is the whole ballgame.

Neural networks are, as a rule, **overconfident**: modern architectures trained to high accuracy
routinely assign 99% probability to predictions that are right 85% of the time. Post-hoc methods
such as temperature scaling or Platt scaling fit a correction on held-out data and are cheap
insurance.

## Conformal prediction

A newer approach worth knowing because it makes a guarantee the others cannot.

**Conformal prediction** wraps any model (trained however you like) and produces intervals with
a *distribution-free coverage guarantee*: if you ask for 90% coverage, you get at least 90%,
without assuming anything about the shape of the errors or the correctness of the model.

The mechanism is simple enough to state in a sentence. Hold out a calibration set the model never
saw, compute the model's absolute errors on it, take the 90th percentile of those errors, and use
it as the interval half-width for new predictions.

```python
residuals = np.abs(y_calib - model.predict(X_calib))
q = np.quantile(residuals, 0.90)
lower, upper = model.predict(X_new) - q, model.predict(X_new) + q
```

The guarantee rests on one assumption: that the calibration data and the new data are
**exchangeable**: drawn from the same distribution. That assumption is exactly what
autocorrelated, non-stationary environmental data tends to violate, so conformal methods for
time series need blocked or time-aware variants. The guarantee is real, but it is not free.

## Reporting uncertainty honestly

Three failure modes, all common in published environmental work:

**Reporting a number with no uncertainty at all.** The most common, and it implicitly claims
infinite precision. If you cannot quantify the uncertainty, say so in words and explain what
would be needed to quantify it.

**Reporting uncertainty that only covers what was easy to compute.** A spread across five
training seeds is not the uncertainty in your prediction; it is the uncertainty from
initialization, which is usually the smallest contributor. Structural uncertainty in the model,
bias in the training data, and uncertainty in the inputs are all typically larger and harder.
State which sources your interval includes and which it does not.

**Burying the uncertainty.** An interval reported in a supplementary table while the abstract
quotes a point estimate has been technically disclosed and practically hidden.

The standard to hold yourself to is the one Assignment 5 asks for explicitly: give the number you
would stand behind, say which validation scheme produced it, and state one thing you could not
determine from the data. That last clause is the one people skip, and it is often the most
useful sentence in the report.

A closing point specific to this field. Where a model's uncertainty is *largest* is frequently
where the stakes are *highest*: the extreme events, the unprecedented conditions, the regions
with no monitoring. The rare cases are rare in the training data too, so the model knows least
about exactly what we most need to know. Any honest treatment of uncertainty in climate
applications has to lead with that, rather than treating it as a caveat at the end.
