# Python review: arrays, dataframes, and plots

This page is a refresher on the handful of operations Assignment 1 actually uses. It is
deliberately short and is not a general Python tutorial — if you want one, the
[Earth Data Science course material](https://earth-ds-ml.github.io/summer_2026/) covers this
ground much more thoroughly.

Everything below assumes the standard three imports:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## NumPy arrays

**Creating them.** `np.arange` gives evenly spaced integers, `np.linspace` gives a fixed
*number* of evenly spaced values between two endpoints. The difference matters: `arange`
takes a step size, `linspace` takes a count.

```python
np.arange(12)                  # 0, 1, 2, ... 11
np.linspace(0, 1, 5)           # 0.0, 0.25, 0.5, 0.75, 1.0  -- exactly 5 values
```

**Shape and reshape.** Every array has a `.shape`, and `reshape` rearranges the same values
into new dimensions. The total number of elements has to stay the same.

```python
a = np.arange(12).reshape(4, 3)
a.shape                        # (4, 3) -- 4 rows, 3 columns
```

**Elementwise operations.** Arithmetic and functions apply to every element at once, with no
loop. This is the main reason NumPy exists.

```python
np.sin(np.linspace(0, np.pi, 5))
a * 2 + 1
```

**Boolean masks.** A comparison produces an array of `True`/`False`, which can then be used
to select. This is how you filter without writing a loop.

```python
mask = a[:, 0] > 3             # compare the first column
a[mask]                        # keep only the rows where the mask is True
```

A useful trick: `True` counts as 1, so `.sum()` on a boolean array *counts* how many
elements satisfy a condition.

```python
(np.sin(x) > 0).sum()          # how many values are positive
```

**The `axis` argument.** This is the one people get wrong. `axis` names the dimension that
gets **collapsed**, not the one that is kept.

```python
a.mean(axis=0)                 # collapses rows -> one value per COLUMN
a.mean(axis=1)                 # collapses columns -> one value per ROW
```

If you are unsure which you want, check the shape of the result against what you expected.

## pandas dataframes

A `DataFrame` is a table with named columns. Columns are accessed by name and behave much
like NumPy arrays.

**Reading a file.** Many scientific data files begin with header comments. `comment="#"`
skips them.

```python
df = pd.read_csv(url, comment="#")
df.head()                      # first five rows -- always look
```

**Filtering.** The same boolean-mask idea, applied to rows. This matters for real data
because missing values are often coded as an impossible number (`-99.99`, `-999`, `-200`)
rather than left blank — and those sentinels will quietly ruin any average you take.

```python
df = df[df["value"] > 0]       # drop rows with a negative sentinel
df["value"].isna().sum()       # count genuine NaNs, column by column
df.describe()                  # count, mean, min, max, quartiles
```

**Dates.** Convert to real datetimes so that plotting and time-based selection work.

```python
df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=15))
df["date"].min(), df["date"].max()
```

**Grouping.** `groupby` splits rows into groups and applies an aggregation to each. It is
how you compute a monthly climatology or an annual mean.

```python
df.groupby("month")["value"].mean()     # one value per calendar month
df.groupby("year")["value"].count()     # how many observations in each year
```

Two related methods worth knowing: `.idxmax()` and `.idxmin()` give the *label* of the
largest and smallest entry rather than the value itself, and `.diff()` gives the change from
one row to the next — which turns a series of annual means into a series of year-over-year
increases.

## Plotting

Use the `fig, ax` form rather than bare `plt.plot`. It is slightly more typing and it scales
to multiple panels without rewriting anything.

```python
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["date"], df["value"], lw=1, label="observed")
ax.set_xlabel("Year")
ax.set_ylabel("Concentration (ppm)")     # ALWAYS include units
ax.set_title("A title that says what is plotted")
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

Every figure in this course needs **axis labels with units and a title**. This is not a
stylistic preference: an unlabelled axis is the most common way for a plot to mislead its
author, not just its reader.

To plot two series together, call `ax.plot` twice before `plt.show()` and add a `label` to
each so the legend can distinguish them. To zoom in on a period, filter the dataframe first
rather than fighting with axis limits:

```python
window = df[(df["year"] >= 2015) & (df["year"] <= 2019)]
```

Other useful calls: `ax.bar` for categorical or per-year values, `ax.scatter` for point
clouds, `ax.axhline(0)` for a reference line, and `ax.set_yscale("log")` when a quantity
spans orders of magnitude.

## Fitting a straight line

`np.polyfit` fits a polynomial of the degree you specify and returns its coefficients,
highest power first. For a straight line that is the slope and the intercept.

```python
slope, intercept = np.polyfit(x, y, 1)
```

Those two numbers *are* the fitted model. You can evaluate them at any inputs you like,
including ones that were not part of the fit — which is what turns a description of past
data into a prediction:

```python
predicted = slope * x_new + intercept
```

Assignment 1 asks you to do exactly this, and then to think carefully about how far outside
the fitted range the result can be trusted.
