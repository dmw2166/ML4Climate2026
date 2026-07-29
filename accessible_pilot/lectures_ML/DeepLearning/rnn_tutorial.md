# Tutorial on Rainfall-Runoff Modeling with RNNs

This hands-on tutorial accompanies the [Recurrent Neural Networks](rnn.md)
notes. Recurrent neural networks were developed in the context of natural
language processing and work well for sequential data; for environmental data
sets, this means they can work well for predicting time series.

In this tutorial, we will train an RNN, LSTM, and GRU model to predict
discharge from streams using the CAMELS data set (Newman et al. 2014). The
CAMELS data set provides 35 years of daily meteorological forcings and
discharge observations from 671 basins across the contiguous United States. As
input to the model, we will use meteorological data (total daily
precipitation, daily min/max temperature, average solar radiation, and vapor
pressure), and we will predict the total discharge from the basin. This is the
**sequence-to-vector** setup from the notes: a full year of past meteorology
in, a single day of discharge out.

This notebook follows a similar example
[here](https://github.com/kratzert/pangeo_lstm_example/blob/master/LSTM_for_rainfall_runoff_modelling.ipynb)
by Frederik Kratzert, implemented in `PyTorch`, which is based off the
following sources:

[1] Kratzert, F., Klotz, D., Brenner, C., Schulz, K., and Herrnegger, M.:
Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks,
Hydrol. Earth Syst. Sci., 22, 6005-6022,
[https://doi.org/10.5194/hess-22-6005-2018](https://doi.org/10.5194/hess-22-6005-2018),
2018a.

[2] Kratzert F., Klotz D., Herrnegger M., Hochreiter S.: A glimpse into the
Unobserved: Runoff simulation for ungauged catchments with LSTMs, Workshop on
Modeling and Decision-Making in the Spatiotemporal Domain, 32nd Conference on
Neural Information Processing Systems (NeuRIPS 2018), Montréal, Canada.
[https://openreview.net/forum?id=Bylhm72oKX](https://openreview.net/forum?id=Bylhm72oKX),
2018b.

[3] A. Newman; K. Sampson; M. P. Clark; A. Bock; R. J. Viger; D. Blodgett,
2014. A large-sample watershed-scale hydrometeorological dataset for the
contiguous USA. Boulder, CO: UCAR/NCAR.
[https://dx.doi.org/10.5065/D6MW2F4D](https://dx.doi.org/10.5065/D6MW2F4D).

:::{admonition} How to use this page (accessible version)
:class: important
On the standard site this is a Jupyter notebook. Here the code is laid out as
a **Python script** you read top to bottom; see [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Like the [CNN tutorial](cnn_tutorial.md), this one **cannot run on an
arbitrary laptop**: it reads the CAMELS data from a requester-pays Google
Cloud Storage bucket (via `gcsfs`) and trains with **TensorFlow**, so it is
meant to be run on a cloud platform such as the **[LEAP
JupyterHub](https://leap.columbia.edu/)** (choose the *TensorFlow* server
image), where training is much faster on a GPU. Every figure is **described**
in a "What the plot shows" block and backed by **printed numbers** (shapes,
parameter counts, losses) you can read. Any plot can also be rendered with
[MAIDR](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/sci_python/trying_maidr.html).
The numbers quoted below are from one real run.
:::

Put these imports and globals at the top of your script.

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide TF's cuFFT/cuDNN/cuBLAS startup logs

from pathlib import Path
from typing import Tuple, List

import gcsfs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# where the CAMELS data lives on Google Cloud
FILE_SYSTEM = gcsfs.core.GCSFileSystem(requester_pays=True)
CAMELS_ROOT = Path('pangeo-ncar-camels/basin_dataset_public_v1p2')
```

## Loading the CAMELS data

This helper function loads the meteorological data for any specific basin from
the CAMELS data set. From the header of the forcing file, we can also extract
the catchment area, which is used to normalize the discharge (to mm/day).

```python
def load_forcing(basin: str) -> Tuple[pd.DataFrame, int]:
    """Load the meteorological forcing data of a specific basin.

    :param basin: 8-digit code of basin as string.

    :return: pd.DataFrame containing the meteorological forcing data and the
        area of the basin as integer.
    """
    # root directory of meteorological forcings
    forcing_path = CAMELS_ROOT / 'basin_mean_forcing' / 'daymet'

    # get path of forcing file
    files = list(FILE_SYSTEM.glob(f"{str(forcing_path)}/**/{basin}_*.txt"))
    if len(files) == 0:
        raise RuntimeError(f'No forcing file file found for Basin {basin}')
    else:
        file_path = files[0]

    # read-in data and convert date to datetime index
    with FILE_SYSTEM.open(file_path) as fp:
        df = pd.read_csv(fp, sep=r'\s+', header=3)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/"
             + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # load area from header
    with FILE_SYSTEM.open(file_path) as fp:
        content = fp.readlines()
        area = int(content[2])

    return df, area
```

This helper function loads the discharge time series for a specific streamflow
basin, converting the units from cubic feet per second to mm/day using the
catchment area.

```python
def load_discharge(basin: str, area: int) -> pd.Series:
    """Load the discharge time series for a specific basin.

    :param basin: 8-digit code of basin as string.
    :param area: int, area of the catchment in square meters

    :return: A pd.Series containing the catchment normalized discharge.
    """
    # root directory of the streamflow data
    discharge_path = CAMELS_ROOT / 'usgs_streamflow'

    # get path of streamflow file
    files = list(FILE_SYSTEM.glob(f"{str(discharge_path)}/**/{basin}_*.txt"))
    if len(files) == 0:
        raise RuntimeError(f'No discharge file found for Basin {basin}')
    else:
        file_path = files[0]

    # read-in data and convert date to datetime index
    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    with FILE_SYSTEM.open(file_path) as fp:
        df = pd.read_csv(fp, sep=r'\s+', header=None, names=col_names)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/"
             + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # normalize discharge from cubic feet per second to mm per day
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10 ** 6)

    return df.QObs
```

We'll load in the data for a single basin and visualize the time series.

```python
basin = '01022500'
df, area = load_forcing(basin)
df['QObs(mm/d)'] = load_discharge(basin, area)

print(df.head())
print(df.columns)
df.info()
```

**What the numbers show.** The DataFrame has one row per day from 1980-01-01
to 2014-12-31 (12784 rows) and 12 columns, all fully populated (no missing
entries; missing discharge is flagged with negative sentinel values instead).
The columns are `Year`, `Mnth`, `Day`, `Hr`, `dayl(s)` (day length),
`prcp(mm/day)` (precipitation), `srad(W/m2)` (solar radiation), `swe(mm)`
(snow water equivalent), `tmax(C)`, `tmin(C)`, `vp(Pa)` (vapor pressure), and
the discharge `QObs(mm/d)` we just added. The first rows (early January 1980)
have zero precipitation, sub-freezing temperatures, and discharge slowly
falling from 1.64 to 1.04 mm/day.

```python
df["QObs(mm/d)"].plot(grid=True, marker=".", figsize=(8, 3.5))
plt.ylabel("Discharge (mm/day)")
plt.xlabel("Year")
plt.show()
```

**What the plot shows.** The full 35-year discharge record. The series is very
spiky: most days sit between 0 and 5 mm/day, but sharp peaks shoot up to
10-20 mm/day every year, and the largest events (around 1989, 1993, 1998, 2005
and 2010) reach 26-28 mm/day. There is a brief dip to negative values at the
very end of the record where missing data is flagged.

```python
df["1995":"2005"]["QObs(mm/d)"].plot(grid=True, marker=".", figsize=(8, 3.5))
plt.ylabel("Discharge (mm/day)")
plt.xlabel("Year")
plt.show()
```

**What the plot shows.** Zooming in on 1995-2005 reveals a seasonal cycle:
discharge rises to clusters of peaks (typically 5-15 mm/day) in the cold and
spring months and drops to near zero each late summer. On top of that cycle
sit occasional extreme events, such as a spike to about 27 mm/day in early
1998. Which years produce extreme peaks is not obvious from the seasonal
pattern alone, which is what makes this a hard prediction problem.

## Preprocess data sets for machine learning

We want to predict the stream discharge rate for the future, given the past
meteorological data sets. We will first preprocess the data using the
`StandardScaler` from `scikit-learn`. The target is the discharge; the
features are five meteorological variables (we drop the date bookkeeping
columns, day length, and snow water equivalent).

```python
targets = df[["QObs(mm/d)"]].copy()
features = df.drop(columns=['Year', 'Mnth', 'Day', 'Hr', 'dayl(s)',
                            'swe(mm)', 'QObs(mm/d)']).copy()

features.hist()
plt.show()
targets.hist()
plt.show()
```

**What the plots show.** Histograms of the five features and the target:

- **prcp(mm/day)**: extremely right-skewed; the vast majority of days have
  near-zero precipitation, with a thin tail out to heavy-rain days.
- **srad(W/m2)**: a broad mound roughly between 100 and 650, peaking near 200.
- **tmax(C)**: spread from about -20 to +35, with a broad plateau above 0.
- **tmin(C)**: spread from about -30 to +20, weighted toward the upper end.
- **vp(Pa)**: right-skewed, from near 0 to about 2500.
- **QObs(mm/d)** (target): strongly right-skewed; most days below 2.5 mm/day
  and a long thin tail out past 15, plus a few negative missing-data flags.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

target_scaler = StandardScaler()
feature_scaler = StandardScaler()

scaled_targets = target_scaler.fit_transform(targets)
scaled_features = feature_scaler.fit_transform(features)

plt.hist(targets)
plt.hist(scaled_targets)
plt.show()
```

**What the plot shows.** Two overlaid histograms: the original discharge
(blue) sits between about -4 and 28, and the scaled discharge (orange) has the
identical shape shifted and squeezed to a mean of 0 and standard deviation of
1, spanning roughly -2.5 to 10. Standard scaling changes the units, not the
shape, so the strong right skew remains. (There may be better scalings for a
variable this skewed, but since the peaks are what we most care about
predicting, we keep it simple here.)

```python
fig, axs = plt.subplots(2, 3, figsize=(8, 6))
ax = axs.ravel()
for i in range(0, 5):
    ax[i].hist(scaled_features[:, i])
    ax[i].set_ylabel(features.columns[i])
plt.tight_layout()
plt.show()
```

**What the plot shows.** The five scaled feature histograms keep their
original shapes but now mostly span about -2 to +3 with means of 0.
Precipitation is the exception: because it is so skewed, its scaled values
run from about -0.5 out to +12.

## Reshaping data for RNN training

We next need to reshape the data into the format the recurrent models expect:
sequential input of shape `(sequence length, number of features)`. We want to
predict a single day of discharge from `n` days of previous meteorological
observations. For `n = 365` and 5 features, a single training sample has
shape `(365, 5)`.

However, the time series is currently one long matrix (days by features). We
need to slide over it and cut out short samples. Keras ships
`tf.keras.utils.timeseries_dataset_from_array` for exactly this, but it fails
on the TensorFlow build in the LEAP image, so we use a short NumPy helper
instead — it takes the same arguments and returns the same
`(inputs, targets)` batches. Here it is on a toy series:

```python
import tensorflow as tf
from tensorflow.keras.models import Model, load_model


def windows(data, targets, sequence_length, batch_size, shuffle=False, seed=None):
    """Slide a window of `sequence_length` over `data`, pairing each with a target."""
    data, targets = np.asarray(data), np.asarray(targets)
    X = np.lib.stride_tricks.sliding_window_view(data, sequence_length, axis=0)
    X = np.moveaxis(X, -1, 1)[:len(targets)]  # (n_windows, sequence_length, n_features)
    ds = tf.data.Dataset.from_tensor_slices((X, targets))
    if shuffle:
        ds = ds.shuffle(len(targets), seed=seed)
    return ds.batch(batch_size)


my_series = [0, 1, 2, 3, 4, 5]
my_dataset = windows(
    my_series,
    my_series[3:],  # the targets are 3 steps into the future
    sequence_length=3,
    batch_size=2
)
print(list(my_dataset))
```

**What the numbers show.** The helper produced two batches. The first batch
pairs the input windows `[0, 1, 2]` and `[1, 2, 3]` with targets `3` and `4`;
the second holds the final window `[2, 3, 4]` with target `5`. This is exactly
the windowing described in the [notes](rnn.md), done for us.

An alternative is the lower-level `window` method on a `tf.data.Dataset`:

```python
for window_dataset in tf.data.Dataset.range(6).window(4, shift=1):
    for element in window_dataset:
        print(f"{element}", end=" ")
    print()

dataset = tf.data.Dataset.range(6).window(4, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window_dataset: window_dataset.batch(4))
for window_tensor in dataset:
    print(f"{window_tensor}")

def to_windows(dataset, length):
    dataset = dataset.window(length, shift=1, drop_remainder=True)
    return dataset.flat_map(lambda window_ds: window_ds.batch(length))
```

**What the numbers show.** The first loop prints windows of length 4 sliding
one step at a time, including the ragged leftovers at the end:

```
0 1 2 3
1 2 3 4
2 3 4 5
3 4 5
4 5
5
```

With `drop_remainder=True` only the three complete windows `[0 1 2 3]`,
`[1 2 3 4]`, `[2 3 4 5]` survive.

## Split data into training, validation, and test data

We also need to designate part of our time series for training, part for
validation, and part for testing. We will use 1980-1995 for training,
1995-2000 for validation, and 2000-2010 as our independent test data set. (The
start dates are in October because of missing data at the beginning of 1980.)

For this stream we don't expect the physics of the catchment to change much
over the period, so splitting by era is acceptable. For a variable strongly
affected by climate change you would have to worry that relationships between
variables shift between the training and test eras.

```python
trainmask = (df.index >= "1980-10-01") & (df.index <= "1995-09-30")
valmask = (df.index >= "1995-10-01") & (df.index <= "2000-09-30")
testmask = (df.index >= "2000-10-01") & (df.index <= "2010-09-30")

trainidx = np.where(trainmask)[0]
validx = np.where(valmask)[0]
testidx = np.where(testmask)[0]

plt.plot(scaled_targets, color="k")
plt.plot(trainidx, scaled_targets[trainidx], color="g", label="train")
plt.plot(validx, scaled_targets[validx], color="r", label="val")
plt.plot(testidx, scaled_targets[testidx], color="b", label="test")
plt.legend()
plt.show()
```

**What the plot shows.** The scaled discharge series colored by split: green
(training) covers the first roughly 5500 points, red (validation) the next
roughly 1800, and blue (test) the following roughly 3650; a little black at
each end is unused. All three segments look statistically similar, with spiky
peaks reaching 8-10 on the scaled axis in each segment (the training segment
even contains some of the highest peaks), so this split introduces no obvious
bias.

We will use an entire year of meteorological data as input to predict the next
time step. Note the target indexing: predictions start at the *end* of the
first 365-day window, so the targets begin `sequence_length - 1` points in.
Getting this alignment right matters. The training set is shuffled (each
window carries its own history, so order across samples doesn't matter); the
test set is not shuffled, so predictions stay in temporal order for
comparison.

```python
sequence_length = 365  # length of the meteorological record given to the network

tf.random.set_seed(42)  # ensures reproducibility

train_ds = windows(
    scaled_features[trainidx],
    targets=scaled_targets[trainidx][sequence_length - 1:],
    sequence_length=sequence_length,
    batch_size=256,
    shuffle=True,
    seed=42
)

valid_ds = windows(
    scaled_features[validx],
    targets=scaled_targets[validx][sequence_length - 1:],
    sequence_length=sequence_length,
    batch_size=2048
)
test_ds = windows(
    scaled_features[testidx],
    targets=scaled_targets[testidx][sequence_length - 1:],
    sequence_length=sequence_length,
    batch_size=len(testidx)
)

for x, y in train_ds.take(1):
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)
```

**What the numbers show.**

```
Input shape: (256, 365, 5)
Target shape: (256, 1)
```

Each training batch holds 256 samples; each sample is 365 days of 5
meteorological features, paired with a single discharge value.

## Train a Simple RNN

We'll first try training an RNN model, with a few hyperparameters shared by
all three models so they can be compared.

```python
import os

cwd = os.getcwd()
model_path = os.path.join(cwd, 'saved_model')

# set some hyperparameters
n_hidden = 10
patience = 20
epochs = 100
learning_rate = 1e-3

tf.random.set_seed(42)  # ensures reproducibility
rnn_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(n_hidden, input_shape=[None, 5]),
    tf.keras.layers.Dense(1)
])
rnn_model.summary()
```

**What the numbers show.** The summary reports a `SimpleRNN` layer with output
shape `(None, 10)` and 160 parameters, plus a `Dense` layer with output shape
`(None, 1)` and 11 parameters: **171 trainable parameters** in total. (These
match the parameter-count formulas in the [notes](rnn.md).) The dense layer is
needed to map the final 10-dimensional hidden state to the single discharge
value we predict.

We'll include a custom metric, the Nash-Sutcliffe Efficiency (NSE), which is a
widely used metric in hydrology for assessing how well a model predicts the
observed data. It is 1 minus the ratio of the squared prediction error to the
variance of the observations: an NSE of 1 is a perfect model, and an NSE of 0
means the model is no better than predicting the mean. The scaler argument
unscales the values first so the NSE is computed on real discharge units.

```python
class NashSutcliffeEfficiency(tf.keras.metrics.Metric):
    def __init__(self, name='nse', scaler=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.sse = self.add_weight(name='sse', initializer='zeros')
        self.sst = self.add_weight(name='sst', initializer='zeros')
        self.scaler = scaler

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.scaler is not None:
            u = self.scaler.mean_
            s = self.scaler.var_
            y_true = y_true*s+u
            y_pred = y_pred*s+u

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        sse = tf.reduce_sum(tf.square(y_true - y_pred))
        sst = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        self.sse.assign_add(sse)
        self.sst.assign_add(sst)

    def result(self):
        return 1.0 - self.sse / self.sst

    def reset_states(self):
        self.sse.assign(0.0)
        self.sst.assign(0.0)
```

We train with the MSE loss (what the optimizer actually minimizes), the Adam
optimizer, and early stopping: if the validation loss stops dropping for
`patience` (20) epochs, training stops and the best weights seen so far are
restored. The NSE is only monitored, not optimized.

```python
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=patience, restore_best_weights=True)
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

rnn_model.compile(loss='mse', optimizer=opt,
                  metrics=[NashSutcliffeEfficiency(scaler=target_scaler)])
history = rnn_model.fit(train_ds, validation_data=valid_ds, epochs=epochs,
                        callbacks=[early_stopping_cb])
rnn_model.save(os.path.join(model_path, 'RNN_timeseries_model.keras'))

valid_loss, valid_nse = rnn_model.evaluate(valid_ds)
print(valid_nse)
```

**What the numbers show.** In our run the RNN trained for the full 100 epochs.
The training loss fell from 1.13 to about 0.25 and the validation loss from
0.90 to about 0.26, while the training NSE rose from -0.34 to about 0.74 and
the validation NSE to about 0.68. Evaluating on the validation set gives an
**NSE of 0.695**.

```python
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean squared error')
plt.plot(history.epoch, np.array(history.history['loss']), label='Train Loss')
plt.plot(history.epoch, np.array(history.history['val_loss']), label='Val loss')
plt.legend()
plt.show()
```

**What the plot shows.** Both loss curves drop steeply from about 1.0 to 0.4
in the first 10 epochs, then decline smoothly and slowly. The validation loss
tracks the training loss closely, flattening out around 0.26 after epoch 60
while the training loss continues slightly lower to about 0.22. The small,
stable gap means mild overfitting at most; there is no rising validation
loss that would signal serious overtraining. The smooth decline also suggests
the learning rate is reasonable (too high and the curve would drop fast then
plateau erratically; too low and it would still be falling at epoch 100).

```python
out = rnn_model.predict(test_ds)
print(out.shape)

for x, y in test_ds.take(1):
    yvals = y.numpy()

plt.plot(yvals, label="true")
plt.plot(out[:, 0], label="prediction")
plt.legend()
plt.show()
```

**What the numbers and plot show.** The prediction has shape `(3288, 1)`: one
value per test day. The plot overlays the true scaled discharge (blue) and the
RNN prediction (orange) over the roughly nine-year test period. The
prediction tracks the overall structure well: it rises and falls with each
seasonal cluster of events. But it consistently underestimates the peaks: the
largest true events reach 8-10 on the scaled axis while the predictions top
out around 4.5. From a flooding perspective the peaks are exactly what we
care most about; extreme values are rare in the training data, which is one
reason data-driven models struggle with them.

## Train an LSTM model

The LSTM should handle the year-long input sequences better, since it is
designed to keep information over longer time ranges. (A dropout layer is
defined here but with `dropout_rate = 0.0` it is inactive; you can raise the
rate to randomly drop a fraction of weights during training as
regularization.)

```python
dropout_rate = 0.0
tf.random.set_seed(42)  # ensures reproducibility
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(n_hidden, input_shape=[None, 5], return_sequences=False),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1)
])
lstm_model.summary()
```

**What the numbers show.** The `LSTM` layer has 640 parameters (four internal
networks of 160, as in the notes), the dropout layer has none, and the dense
layer adds 11: **651 trainable parameters**, nearly four times the RNN.
`return_sequences=False` means the model outputs only the final hidden state
(we predict one target day, not a sequence).

```python
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=patience, restore_best_weights=True)
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

lstm_model.compile(loss='mse', optimizer=opt,
                   metrics=[NashSutcliffeEfficiency(scaler=target_scaler)])
history_lstm = lstm_model.fit(train_ds, validation_data=valid_ds, epochs=epochs,
                              callbacks=[early_stopping_cb])
lstm_model.save(os.path.join(model_path, 'LSTM_timeseries_model.keras'))

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean squared error')
plt.plot(history.epoch, np.array(history.history['loss']), label='Train Loss - RNN')
plt.plot(history.epoch, np.array(history.history['val_loss']), label='Val loss - RNN')
plt.plot(history_lstm.epoch, np.array(history_lstm.history['loss']), label='Train Loss - LSTM')
plt.plot(history_lstm.epoch, np.array(history_lstm.history['val_loss']), label='Val loss - LSTM')
plt.legend()
plt.show()
```

**What the numbers and plot show.** The LSTM also ran 100 epochs in our run,
ending with a training loss near 0.07 and validation loss near 0.13-0.14
(validation NSE about 0.83-0.84). On the comparison plot the two LSTM curves
drop faster and sit well below both RNN curves from about epoch 10 onward:
LSTM validation loss levels off around 0.13-0.14 versus 0.26 for the RNN,
roughly half the error. A modest gap opens between the LSTM training and
validation curves late in training, hinting at slight overfitting, but the
validation loss is not rising. The LSTM's advantage is consistent with it
capturing longer-range dependencies across the 365-step input that the RNN
forgets.

```python
out_lstm = lstm_model.predict(test_ds)

plt.plot(yvals, label="true")
plt.plot(out_lstm[:, 0], label="prediction - LSTM")
plt.legend()
plt.show()
```

**What the plot shows.** True test discharge (blue) against the LSTM
prediction (orange). Compared with the RNN, the LSTM reaches much higher into
the peaks: several predicted events now reach 5-6 on the scaled axis, and
mid-sized events are matched closely. The very largest events (8-10) are
still underestimated. Getting more of the extremes right would probably take
more training data (for example, training across many basins rather than
one), oversampling extreme events, or extra input features such as snowpack.

## Train a GRU model

The GRU sits between the RNN and the LSTM in complexity. (Keras reports
slightly more parameters than the classic GRU formula from the notes because
its default implementation, `reset_after=True`, carries a second set of bias
vectors.)

```python
tf.random.set_seed(42)  # ensures reproducibility
gru_model = tf.keras.Sequential([
    tf.keras.layers.GRU(n_hidden, return_sequences=False, input_shape=[None, 5]),
    tf.keras.layers.Dense(1)
])
gru_model.summary()
```

**What the numbers show.** The `GRU` layer has 510 parameters plus the same
11-parameter dense layer: **521 trainable parameters**, between the RNN's 171
and the LSTM's 651.

```python
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=patience, restore_best_weights=True)
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

gru_model.compile(loss='mse', optimizer=opt,
                  metrics=[NashSutcliffeEfficiency(scaler=target_scaler)])
history_gru = gru_model.fit(train_ds, validation_data=valid_ds, epochs=epochs,
                            callbacks=[early_stopping_cb])
gru_model.save(os.path.join(model_path, 'GRU_timeseries_model.keras'))

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean squared error')
plt.plot(history.epoch, np.array(history.history['loss']), label='Train Loss - RNN')
plt.plot(history.epoch, np.array(history.history['val_loss']), label='Val loss - RNN')
plt.plot(history_lstm.epoch, np.array(history_lstm.history['loss']), label='Train Loss - LSTM')
plt.plot(history_lstm.epoch, np.array(history_lstm.history['val_loss']), label='Val loss - LSTM')
plt.plot(history_gru.epoch, np.array(history_gru.history['loss']), label='Train Loss - GRU')
plt.plot(history_gru.epoch, np.array(history_gru.history['val_loss']), label='Val loss - GRU')
plt.legend()
plt.show()
```

**What the numbers and plot show.** In our run the GRU stopped early at epoch
65 (its validation loss had stopped improving for 20 epochs), ending with a
training loss near 0.11 and validation loss near 0.18-0.20 (validation NSE
about 0.78). On the six-curve comparison the GRU curves fall between the RNN
and the LSTM, tracking the LSTM closely for the first 25 epochs before
leveling off slightly higher. Final validation losses in this run: RNN about
0.26, GRU about 0.18, LSTM about 0.13.

```python
out_gru = gru_model.predict(test_ds)

plt.plot(yvals, label="true")
plt.plot(out_lstm[:, 0], label="prediction - LSTM")
plt.plot(out_gru[:, 0], label="prediction - GRU")
plt.plot(out[:, 0], label="prediction - RNN")
plt.legend()
plt.show()
```

**What the plot shows.** All three predictions overlaid on the true test
series. The LSTM and GRU traces sit close together and reach clearly higher
into the peaks than the RNN trace; the RNN is visibly the flattest. All three
still miss the very largest events (the true series tops out near 10 on the
scaled axis; the best predictions reach about 6). Ranking the models on this
run: LSTM best, GRU close behind, vanilla RNN last, matching the expectations
from the [notes](rnn.md).

## Where to go from here

We trained on a single basin for speed, but the authors of the source notebook
explicitly recommend against that for real applications: models trained on
**many basins at once** (the catchment-independent approach) are both better
and more generalizable, because hundreds of streams provide many more examples
of rare high-flow events. Kratzert and colleagues have continued this line of
work, including expanding CAMELS into the multi-country **Caravan** data set.
Other directions to explore: oversampling or reweighting the extreme events,
adding input features such as snowpack, deeper or wider recurrent models
(with the attendant overfitting risk), and better ways of splitting a
temporal record than a single chronological cut.
