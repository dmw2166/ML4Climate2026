# Assignment 7: Using Unsupervised Machine Learning to Discover Climate Zones

:::{admonition} How to do this assignment (accessible version)
:class: important
On the standard site this is a Jupyter notebook with empty code cells to fill in.
Here, do it as a **Python script** — see [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Create a file `assignment3.py`, write your answer to each numbered question in its
own labelled section (a `# Q1`, `# Q2`, ... comment helps you navigate), and run it
with `python assignment3.py`. Turn in the `.py` script as described in Assignment 1,
Part 6.

Most of what this assignment asks you to "plot" is a **map** (a value at each
latitude/longitude point) or a **cluster map** (a colour label at each point).
Those are spatial pictures that a scatter/`imshow` renders visually. Instead of
relying on the picture, answer each question with **text summaries you print**:
array shapes, min/max/mean values, and — for the clustering parts — how many
points fall in each cluster (`np.bincount(labels)`) and roughly *where* each
cluster sits (mean latitude/longitude of its points). You can still render any
plot through [MAIDR](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/sci_python/trying_maidr.html)
if you find it useful, but the printed numbers are what you discuss in your answer.
:::

In this homework assignment, you will explore some common clustering methods, including
K-Means clustering and Gaussian Mixture Models.

You'll apply these clustering algorithms to the problem of classifying climate
zones over the continental United States. The Köppen-Geiger Climate Zones are a
climate classification system based on precipitation and temperature
([NOAA description](https://www.noaa.gov/jetstream/global/climate-zones/jetstream-max-addition-k-ppen-geiger-climate-subdivisions)).
You'll use unsupervised machine learning to discover similar climate zones using
the climatological averages for temperature and precipitation.

```python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
```

## Download the data sets

You'll use climatological data over the continental US, specifically average
monthly precipitation and temperature records for the years 1901 to 2000 from
NOAA ([source](https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals)).
Download the two NetCDF files by running these `wget` commands **in a terminal**
(not in the script) from the folder where your script lives:

```
wget "https://www.ncei.noaa.gov/data/oceans/archive/arc0196/0248762/1.1/data/0-data/tavg-1901_2000-monthly-normals-v1.0.nc"
wget "https://www.ncei.noaa.gov/data/oceans/archive/arc0196/0248762/1.1/data/0-data/prcp-1901_2000-monthly-normals-v1.0.nc"
```

## Part 1: Load and prepare the climatological data

**1)** Use `xarray` to open the NetCDF files for the climatological temperature
and precipitation datasets for the continental US. *(Hint: `xr.open_dataset(...)`.
Then `print(tavg)` to read the list of variables, dimensions, and coordinates as
text — that text is the accessible equivalent of "looking at" the file.)*

**2)** Make a plot of the monthly average precipitation in January over the
continental US. *(This is a map. Instead of reading the picture, also print a text
summary of the January precipitation field: its shape, and its min, max, and mean
over all land points — e.g. `np.nanmin`, `np.nanmax`, `np.nanmean`. Note the units
from the variable's metadata in step 1.)*

**3)** Make a plot of the monthly average temperature in August over the
continental US. *(Same as above — print shape, min, max, and mean of the August
temperature field alongside any plot.)*

To identify climatological zones across the continental US, you will use the 12
monthly average temperature values and the 12 monthly average precipitation values
as input features for clustering algorithms. Each latitude and longitude point will
be treated as a single sample with 24 features (12 for temperature and 12 for
precipitation).

**4)** Extract the values for `"mlytavg_norm"` from the temperature and
precipitation NetCDF files and store them in NumPy arrays named `avgtemp` and
`avgprec`, respectively. *(Print `avgtemp.shape` and `avgprec.shape` — you should
see a month dimension of length 12 and two spatial dimensions.)*

**5)** Put the latitude and longitude values into numpy arrays, and use the
`np.meshgrid` function to create 2D arrays giving the latitude and longitude value
at each point on the map:

```python
lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
```

You'll use these arrays later to create labels for the latitude and longitude points
associated with each sample. *(Print `lat_grid.shape` to confirm it matches one
month of the temperature map.)*

**6)** Use the `np.isnan` function to create a mask that is `True` where there is
data and `False` where there is no data (ocean / outside the US) in the 2D
temperature and precipitation maps. This mask should have the same dimensions as
latitude by longitude. *(Hint: `np.isnan` returns `True` where a value is `NaN`, so
`~np.isnan(one_month_map)` is `True` where data exists. Print
`mask.sum()` — the number of valid land points — this is how many samples you'll
cluster.)*

**7)** Use the mask to index the numpy arrays `avgtemp`, `avgprec`, `lat_grid`, and
`lon_grid` to create arrays called `avgtemp_masked`, `avgprec_masked`, `lat_masked`,
and `lon_masked`. These arrays should no longer contain any NaN's. *(Print
`avgtemp_masked.shape`. Check `np.isnan(avgtemp_masked).sum()` prints `0`.)*

**8)** `scikit-learn` functions assume that the sample number ($n_{sample}$) is the
first dimension of an array, and the features associated with a sample are the
second dimension. Transpose `avgprec_masked` and `avgtemp_masked` to get the correct
ordering of dimensions. Check that the shape of the arrays is now
$n_{sample} \times 12$. *(Print the shapes to confirm.)*

## Part 2: Pre-process the data

**9)** Scale the precipitation and temperature arrays between -1 and 1. You do this
so that all 12 months are scaled relative to the same minimum and maximum values
for precipitation or temperature, respectively. You can use the `MinMaxScaler` from
scikit-learn with `feature_range=(-1, 1)` (you will have to reshape the arrays to a
single column to do this, then reshape back), or write your own scaling function.
*(Print the min and max of each scaled array — they should be -1 and 1.)*

**10)** Create a feature array `X` that is $n_{samples} \times n_{features}$, where
$n_{features} = 24$ (i.e. it combines the two arrays that contain the scaled
temperature and precipitation monthly averages associated with each sample).
*(Hint: `np.concatenate([...], axis=1)`. Print `X.shape` — it should be
$n_{samples} \times 24$.)*

## Part 3: Use K-Means Clustering to Label Climate Zones

**11)** Use the `KMeans` method from `sklearn.cluster` to fit 8 clusters to the `X`
feature matrix. *(Set `random_state=0` so your result is reproducible. The cluster
label for each sample is in `kmeans.labels_`.)*

**12)** Make a scatter plot of `lat_masked` vs. `lon_masked` and colour each point
by its K-Means label, using point size 0.1 and a **discrete** colormap (e.g.
`cmap=plt.get_cmap("tab10", 8)`).

Because this is a spatial map, describe it in **text** instead of relying on the
picture. Print, for each of the 8 clusters: how many points it contains, and the
mean latitude and longitude of those points (roughly *where* on the map that
climate zone sits):

```python
for k in range(8):
    pts = kmeans.labels_ == k
    print(f"cluster {k}: {pts.sum():5d} points, "
          f"mean lat {lat_masked[pts].mean():.1f}, "
          f"mean lon {lon_masked[pts].mean():.1f}")
```

You can compare the climate zones discovered by the K-Means clustering approach
with the map [here](https://www.cec.org/mapmonday/climate-zones-in-north-america/).
In your answer, describe which cluster corresponds to which broad region (e.g. the
arid Southwest, the humid Southeast, the cold northern interior) using the mean
latitude/longitude you printed.

**13)** Repeat parts 11 and 12 but choose a **different** number of clusters. Print
the same per-cluster summary. In your answer, discuss how the discovered zones
change — do more clusters split one region into finer sub-zones, or split it in a
way that doesn't match real climate boundaries?

## Part 4: Use a Gaussian Mixture Model to Label Climate Zones

**14)** Use the `GaussianMixture` method from `sklearn.mixture` to fit 8 components
to the `X` feature matrix. *(Set `random_state=0`. Get the label for each sample
with `gmm.predict(X)`.)*

**15)** Make a scatter plot of `lat_masked` vs. `lon_masked` and colour by the
components learned by the Gaussian Mixture Model, using point size 0.1 and a
discrete colormap.

As in question 12, describe the result in **text**: print the number of points and
the mean latitude/longitude for each of the 8 GMM components. In your answer,
compare the GMM zones to the K-Means zones from Part 3 — where do they agree, and
where does allowing non-spherical clusters (GMM) change which points group
together?