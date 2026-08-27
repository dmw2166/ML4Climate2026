# Assignment 2: Data Pre-Processing of Historical Tropical Cyclone Records

:::{admonition} How to do this assignment (accessible version)
:class: important
On the standard site this is a Jupyter notebook with empty code cells to fill in.
Here, do it as a **Python script** — see [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Create a file `assignment1.py`, write your answer to each numbered question in its
own labelled section (a `# Q1`, `# Q2`, ... comment helps you navigate), and run it
with `python assignment1.py`. For any plot a question asks for, render it through
**MAIDR** so you can hear/read it, and also print a text summary (`describe()`,
`value_counts()`, `.shape`) — you can discuss those in your answer. Turn in the
`.py` script (and any saved `.npy` arrays) as described in Part 6.
:::

In this assignment, you will explore how to do data exploration and pre-processing
with a .csv file containing a global collection of tropical cyclone records, the
International Best Track Archive for Climate Stewardship
[(IBTrACS)](https://www.ncei.noaa.gov/products/international-best-track-archive).
The column variable descriptions are given
[here](https://www.ncei.noaa.gov/sites/default/files/2025-09/IBTrACS_v04r01_column_documentation.pdf).

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## Load and aggregate the data set

Using the following code, load in the data set (this takes a few seconds to run):

```python
url = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv'
df = pd.read_csv(url, parse_dates=['ISO_TIME'], usecols=range(12),
                 skiprows=[1], na_values=[' ', 'NOT_NAMED'],
                 keep_default_na=False, dtype={'NAME': str})
```

This data set includes cyclone tracks (so it has multiple entries per named
cyclone). We'll use the code below to create an aggregated data set for only the
named cyclones, which has one entry per cyclone. You will use the data set
`dfnamed` for the rest of the assignment.

```python
dfnamed = df.groupby("NAME").agg(MAX_WIND=('WMO_WIND', 'max'),
                                 MIN_PRES=('WMO_PRES', 'min'),
                                 MEAN_LAT=('LAT', 'mean'),
                                 MEAN_LON=('LON', 'mean'),
                                 BASIN=('BASIN', 'first'),
                                 SUBBASIN=('SUBBASIN', 'first'),
                                 NATURE=('NATURE', 'first'),
                                 SEASON=('SEASON', 'first')).reset_index()
dfnamed.head()

# these lines of code remove the initial dataframe (we won't need it anymore)
import gc
del df
gc.collect()
```

## Part 1: Data Exploration

How many named cyclones are there?

```python
len(dfnamed)
```

**1)** Use the `pandas` `hist` method to plot the marginal distributions of the
variables in the dataframe `dfnamed`. *(Render with MAIDR to explore each
distribution, and/or print `dfnamed.describe()` to summarize them in text.)*

**2)** Use the `seaborn` `PairGrid` function to create a scatterplot of all of the
variables. *(A pair plot is dense; also print `dfnamed.corr(numeric_only=True)` as
a text table so you can discuss which variables move together.)*

**3)** Using `matplotlib`, create a scatter plot of the minimum pressure vs. the
maximum wind speed and color by the year the cyclone took place (`SEASON`). *(This
is a scatter plot — MAIDR can sonify it. Expect a strong negative relationship:
stronger storms have lower minimum pressure.)*

## Part 2: Handle missing data

**4)** How many non-null values does each variable have? *(Hint: `dfnamed.info()`
or `dfnamed.notna().sum()` — both print a text table.)*

**5)** Create a new dataframe called `dfdrop` where you have discarded the rows
with NaN values in `dfnamed`.

**6)** Create a new dataframe called `dfimputed` where you have imputed the missing
values in `dfnamed` with 0.0.

## Part 3: Feature scaling

**7)** Scale the `MAX_WIND` column of your `dfdrop` dataframe using the
`StandardScaler` and put it into an array called `y`, since it's the target we want
to predict.

*Hint*: the `StandardScaler` expects a 2-D input, so reshape your column with
`.values.reshape(-1, 1)` before scaling.

**8)** Create a copy of your `dfdrop` dataframe called `dffeatures`, and drop the
`NAME` and `MAX_WIND` columns from your `dffeatures` dataframe, since the
`MAX_WIND` variable is going to be your target variable and we won't need the
cyclone names any longer.

## Part 4: Encode Categorical Variables

**9)** Check which unique categories each of the variables `BASIN`, `SUBBASIN`, and
`NATURE` take, using the `dffeatures` dataframe.

**10)** Print out the number of cyclones in each basin and subbasin. Also print out
how many of each storm type there is. *(Hint: `value_counts()` on each column.)*

**11)** Encode the `BASIN`, `SUBBASIN`, and `NATURE` variables in `dffeatures`
using One Hot Encoding, and standardize `MIN_PRES`, `MEAN_LAT`, `MEAN_LON`, and
`SEASON` using the `StandardScaler`.

*Hint*: you can do this in one step using the `sci-kit learn` `ColumnTransformer`.
Create an array called `X` that contains the encoded categorical variables and the
scaled numerical variables.

**12)** Print out the feature names associated with the columns in your `X` array.
*(Hint: `get_feature_names_out()` on your fitted `ColumnTransformer`.)*

## Part 5: Train, Validation, & Test Split

**13)** Split your data set into a training and test/validation data set using the
`train_test_split` function with 80% of your original data for training and 20% for
the testing and validation data sets.

**14)** Split your `X_test_val` and `y_test_val` again into separate validation and
test data sets, that are 10% each of the original data set. Double check that the
size of your final training, validation, and test data sets are correct by printing
out the shape of each array.

**15)** Save the training, validation, and test data sets and labels as numpy
arrays using `np.save()`.

## Part 6: Create your Assignments Repository

To turn in this and other homework assignments for this course, you will create an
assignments `github` repository. Do this in the accessible terminal (Git Bash in VS
Code) set up in the [accessible workflow
guide](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html);
the [Intro to
Git](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/intro_to_git.html)
lesson from CLMT5045 covers each command.

- Create a new directory called `ml4climate2026` in your home directory.
- Create a `Readme.md` markdown file that contains your name.
- Initialize a new git repository.
- Add the file and make your first commit.
- Create a new **private** repository on Github called `ml4climate2026`. (Call it
  exactly like that. Do not vary the spelling, capitalization, or punctuation.)
- Push your `ml4climate2026` repository to Github.
- On Github, go to "settings" → "collaborators", and add `dmw2166`.
- Push new commits to this repository whenever you are ready to hand in your
  assignments (including your `assignment1.py` script and saved `.npy` arrays).