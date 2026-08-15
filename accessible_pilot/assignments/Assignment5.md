# Assignment 5: Flood Risk Prediction with Neural Network Regression

:::{admonition} How to do this assignment (accessible version)
:class: important
On the standard site this is a Jupyter notebook with empty code cells to fill
in. Here, do it as a **Python script**; see [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Create a file `assignment4.py`, write your answer to each numbered question in
its own labelled section (a `# Q1`, `# Q2`, ... comment helps you navigate),
and run it with `python assignment4.py`. Turn in the `.py` script as described
in Assignment 1, Part 6.

Two of the questions ask for plots (a histogram grid and a scatter plot).
Instead of relying on the pictures, answer them with **text summaries you
print**: `df.describe()` for the histograms, and correlation/error numbers for
the scatter plot; the question text below says exactly what to print. You can
still render any plot through
[MAIDR](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/sci_python/trying_maidr.html)
if you find it useful, but the printed numbers are what you discuss in your
answer.

This assignment trains a small fully connected neural network with
**TensorFlow**. Unlike the CNN and RNN tutorials, it runs fine on an ordinary
laptop CPU: the dataset is a single CSV file and the model is small. Install
what you need with `pip install kagglehub tensorflow scikit-learn pandas`.
:::

For this assignment you will develop a machine learning model to predict the
probability of a flood given both environmental and social factors. The data
set comes from the Kaggle [Flood Prediction
Dataset](https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset).

Download the data with `kagglehub`:

```python
import os
import kagglehub

# Download latest version
path = kagglehub.dataset_download("naiyakhalid/flood-prediction-dataset")

print("Path to dataset files:", path)
print(os.listdir(path))
```

The folder contains a single file, `flood.csv`.

## Part 1: Load and Preprocess the Flood Data Set

**1)** Load in the flood data set using `pandas`. *(Print `df.shape`,
`df.columns`, and `df.head()`. Read the column list: there are 20 predictor
columns describing environmental and social factors, for example monsoon
intensity, deforestation, urbanization, dams, and drainage, plus the
`FloodProbability` column you will predict.)*

**2)** Make a histogram plot of the different numerical variables in the
dataset. *(The accessible equivalent: print `df.describe()` and read the
count, mean, standard deviation, min, and max of each column. In your answer,
note the ranges: the predictors are small non-negative integer scores on
similar scales, and `FloodProbability` is a fraction between roughly 0.3 and
0.7.)*

**3)** Create the feature matrix and the targets vector. The target will be
the flood probability, and the predictors will be all of the other variables
in the data frame. Put the flood probability into a `numpy` array called `y`
and the other variables into a `numpy` array called `X`. *(Print `X.shape` and
`y.shape`: the number of rows should match the DataFrame, with 20 feature
columns in `X`.)*

## Part 2: Preprocessing

**4)** Use the `StandardScaler` method to scale the numerical variables in the
`X` and `y` arrays. *(Keep the fitted scalers; you will need the target scaler
again in question 14 to unscale predictions. `StandardScaler` expects 2-D
input, so reshape `y` to a single column if needed. Print the mean and
standard deviation of one scaled column to confirm they are about 0 and 1.)*

## Part 3: Training, validation, and test splits

**5)** Split the data into training, validation, and test data sets, with 80%
of the data used for training, and 10% each for validation and testing.
*(Hint: call scikit-learn's `train_test_split` twice: first to set aside 20%,
then to split that 20% in half. Print the shape of all six arrays and check
the row counts are in the ratio 80/10/10.)*

## Part 4: Train a Neural Network

**6)** Using `tensorflow`, create a fully connected neural network that takes
as input a feature matrix of size 20, and has 3 dense layers with 100 neurons
and `ReLU` activation and a final dense layer (with no activation) to get to a
single output value. *(This is the plain ANN from the
[deep learning notes](../lectures_ML/DeepLearning/ann.md): no activation on
the final layer because this is regression.)*

**7)** Print off a summary of the model. How many total trainable parameters
does it have? *(`model.summary()` prints a table with one row per layer and a
total at the end; state the total in your answer. Screen-reader tip: the table
is drawn with box characters, so it can be easier to read the per-layer
counts with `for w in model.trainable_weights: print(w.shape)` and sum them.)*

**8)** Compile the model with MSE loss and use the `Adam` optimizer. As a
metric, include the Root Mean Squared Error. *(Hint:
`metrics=[tf.keras.metrics.RootMeanSquaredError()]`. As in the RNN tutorial,
the loss is what the optimizer minimizes; metrics are just monitored.)*

## Train and Evaluate the Model

**9)** Train the model for 30 epochs on the training data set you created
earlier, using the validation data set to validate the model. *(Pass
`validation_data=(X_val, y_val)` to `model.fit` and keep the returned
`history` object.)*

**10)** Make a plot of the training and validation loss vs. epoch number.
*(The numbers behind the plot are in `history.history["loss"]` and
`history.history["val_loss"]`. Print the first epoch's and last epoch's value
of each, and in your answer describe the curve in words: how fast the loss
falls, whether the validation loss tracks the training loss, and whether it
starts rising at the end, which would signal overfitting.)*

**11)** Save the trained model. *(For example
`model.save("flood_model.keras")`.)*

**12)** Evaluate the model on the test data set (i.e. print off the MSE loss
and the Root Mean Squared Error). *(`model.evaluate(X_test, y_test)` returns
both numbers; remember these are in scaled units.)*

**13)** Get the model predictions for the test data set. *(Print the shape of
the prediction array: one value per test sample.)*

**14)** Unscale both `y_test` and the model predictions for the flood
probabilities by using the inverse transformation for the standard scaler.
*(Hint: `target_scaler.inverse_transform(...)`. Print the min and max of the
unscaled predictions and check they land in the plausible flood-probability
range you saw in question 2.)*

**15)** Make a scatter plot of the predicted flood probabilities compared with
the true flood probabilities for the test data set. *(The accessible
equivalent of judging the scatter by eye: print the correlation coefficient
between predicted and true values, `np.corrcoef(y_true, y_pred)[0, 1]`, the
RMSE in original probability units, and a handful of (true, predicted) pairs,
for example the first ten. In your answer, describe how tightly the
predictions track the truth: a correlation near 1 and points close to the
one-to-one line mean the network has learned the mapping well.)*
