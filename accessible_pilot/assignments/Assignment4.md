# Assignment 4: Predicting Health Impacts from Air Quality Factors

:::{admonition} How to do this assignment (accessible version)
:class: important
On the standard site this is a Jupyter notebook with empty code cells to fill in.
Here, do it as a **Python script** — see [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Create a file `assignment2.py`, write your answer to each numbered question in its
own labelled section (a `# Q1`, `# Q2`, ... comment helps you navigate), and run it
with `python assignment2.py`. For any plot a question asks for, render it through
**MAIDR** so you can hear/read it, and also print a text summary (`describe()`,
`value_counts()`, `.shape`) — you can discuss those in your answer. Turn in the
`.py` script as described in Assignment 1.
:::

In this assignment, we will use a (synthetic) data set looking at how health
impacts are related to air quality factors. This data set is available at
[https://www.kaggle.com/datasets/rabieelkharoua/air-quality-and-health-impact-dataset](https://www.kaggle.com/datasets/rabieelkharoua/air-quality-and-health-impact-dataset).
The data includes public health outcomes and how they are related to air quality
and meteorological factors.

```python
import pandas as pd
import matplotlib.pyplot as plt
import os
```

## Download the data from Kaggle

To download data from Kaggle, install the `kagglehub` package once in the terminal
(not in the script): `pip install kagglehub`. Then download the data set:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("rabieelkharoua/air-quality-and-health-impact-dataset")
print("Path to dataset files:", path)

os.listdir(path)
```

## Part 1: Data exploration

**1)** Load in the csv file as a dataframe using `pandas`.

**2)** Check whether there are any NaN's in the dataframe. *(Hint: `df.isna().sum()`
prints a text table of missing values per column.)*

**3)** Make a histogram of the different numerical variables in the dataframe.
*(Render with MAIDR to explore each distribution, and/or print `df.describe()` to
summarize them in text.)*

**4)** There are two possible targets in the dataframe. One is a categorical
variable `HealthImpactClass`, and the other is a numerical variable
`HealthImpactScore`. Create numpy arrays, one named `y_classification` containing
the `HealthImpactClass`, and one named `y_regression` containing the
`HealthImpactScore`.

**5)** Check how balanced the 5 classes are in `HealthImpactClass`. *(Hint:
`value_counts()` prints the count per class as text.)*

**6)** Create a numpy array called `features` that includes the following 9
variables:

- AQI
- PM10
- PM2_5
- NO2
- SO2
- O3
- Temperature
- Humidity
- WindSpeed

## Part 2: Preprocessing

**7)** Create two python lists, one including the class names, and the other
including the feature names. The classification of health impact is derived from
the health impact score, using the following thresholds:

- 0: 'Very High' (HealthImpactScore >= 80)
- 1: 'High' (60 <= HealthImpactScore < 80)
- 2: 'Moderate' (40 <= HealthImpactScore < 60)
- 3: 'Low' (20 <= HealthImpactScore < 40)
- 4: 'Very Low' (HealthImpactScore < 20)

**8)** Use the `StandardScaler` method to scale the numerical variables in the
`features` array, and save this as a numpy array `X`.

## Part 3: Training, validation, and test split

**9)** Split the data into training, validation, and test data sets, with 80% of
the data used for training and 10% each for validation and testing. Create separate
regression and classification targets, using `y_classification` and `y_regression`.

## Part 4: Train a Random Forest Classifier

**10)** Train a `RandomForestClassifier` with 120 estimators and a maximum depth of
10. Set the `class_weight` to `"balanced"`, since the classes are imbalanced. You
can use the default values for the other hyperparameters.

**11)** Report the confusion matrix showing the performance of the trained
classifier on the validation data set. On the standard site this is drawn as a
labelled heatmap; in the accessible workflow, **print the matrix as numbers**
instead, with the class names, so you can read it row by row:

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_classification_val, clf.predict(X_val))
print("Rows = true class, Cols = predicted class, order =", classnames)
print(cm)
# A full per-class precision/recall/F1 table:
from sklearn.metrics import classification_report
print(classification_report(y_classification_val, clf.predict(X_val), target_names=classnames))
```

Each row of the matrix is a true class and each column is a predicted class; the
diagonal entries are correct predictions. To explore it visually with sound, you
can also render the confusion matrix as a heatmap through MAIDR.

Because the classes are imbalanced, the confusion matrix shows that the classifier
does not perform that well on the classes that are not well-represented in the
data. One way to improve this is to use over-sampling to augment the data set.
Using the `imbalanced-learn` library (install once with `pip install
imbalanced-learn`), we can use the `SMOTE` algorithm
([https://arxiv.org/pdf/1106.1813](https://arxiv.org/pdf/1106.1813)) to oversample
the data set:

```python
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_classification_train)
```

**12)** Train a new random forest classifier using `X_resampled` and `y_resampled`.
Use the same hyperparameters as your original random forest.

**13)** Report the confusion matrix (as numbers, per Q11) for the classifier that
was trained on the oversampled data set, evaluated on the validation data set.
Compare it with the matrix from Q11 — which classes improved?

## Part 5: Train a Random Forest Regressor

**14)** Train a `RandomForestRegressor` on the training data set, using the
regression target. Set the number of estimators to 120 and the maximum tree depth
to 10. You can use the other default hyperparameters.

**15)** Using your trained `RandomForestRegressor`, predict the target values for
the validation data set and calculate the coefficient of determination (R²) between
the true targets and the predicted values. *(Hint: `sklearn.metrics.r2_score`, or
the regressor's `.score()` method, prints a single number.)*

**16)** Report the feature importance of your trained random forest. On the
standard site this is a bar plot; in the accessible workflow, print it as a sorted
text list so you can read the ranking:

```python
import pandas as pd
print(pd.Series(reg.feature_importances_, index=featurenames).sort_values(ascending=False))
```

You can also render the bar plot through MAIDR if you want to explore it by sound.

**17)** What are the 4 most important features in terms of determining the health
impact score? Print them out. *(The sorted list from Q16 gives them directly — take
the top 4 names.)*
