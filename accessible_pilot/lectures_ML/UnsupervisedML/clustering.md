# Clustering

:::{admonition} How to work through these notes (accessible version)
:class: important
On the standard site this page is a Jupyter notebook. Here, the code is written to be
run as a **Python script** in VS Code — see [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Put the code into a file such as `clustering.py`, adding the sections below one at a
time, and run it with `python clustering.py`. Read printed output with the terminal's
Accessible View.

**Plots.** Each figure has a **"What the plot shows"** description. Clustering is an
unusually visual topic — the whole point is often "look at the shape of the data" — so
throughout this page the code **prints a numerical version of each argument**, using
scores that measure how well a clustering matches the true grouping. You should never
need the picture to follow the reasoning. To explore a plot yourself with sound and a
text/braille readout, use **MAIDR** — see [Accessible plots with
MAIDR](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/sci_python/trying_maidr.html).
:::

**Clustering** algorithms group data into clusters based on **similarity**, without
predefined labels. This is the second major branch of classical unsupervised machine
learning, alongside the [dimensionality reduction](dimensionalityreduction.md) methods we
have just covered.

These notes cover:

- clustering versus supervised classification, and what clustering is good for
- the landscape of clustering algorithms, and their trade-offs
- **K-means clustering** — the algorithm, and its failure modes
- how to **choose the number of clusters** (inertia and the elbow method, the silhouette
  score)
- **DBSCAN**
- **Gaussian Mixture Models**

## Clustering vs. supervised classification

The contrast with supervised classification is the place to start. In classification we
already have labels, and we are predicting them based on known outcomes. In clustering we
have **no labels a priori**, and we are trying to discover the inherent patterns and
relationships in the data.

```python
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, 2:4]      # petal length, petal width
y = iris.target

# What a supervised model gets: the labels.
print("petal length / petal width, by known species:")
for k, name in enumerate(iris.target_names):
    pts = X[y == k]
    print(f"  {name:12s}: length {pts[:,0].mean():.2f} +/- {pts[:,0].std():.2f}, "
          f"width {pts[:,1].mean():.2f} +/- {pts[:,1].std():.2f}")

# What a clustering algorithm gets: just the points, no labels.
print(f"\nWhat clustering sees: {X.shape[0]} unlabeled points in 2 dimensions.")
```

:::{admonition} What the plot shows
:class: note
Two scatter plots side by side, both of petal width against petal length.

**Left — supervised classification**: the points are colored by their known species.
*Setosa* forms a small, tight group in the bottom-left, far from everything else.
*Versicolor* and *virginica* sit in the upper right, adjacent to each other and touching
along a boundary.

**Right — clustering**: exactly the same points, all in a single uniform gray. No color,
no groups marked.

That right-hand panel is all a clustering algorithm ever gets. The bottom-left group is
obviously separate, and any method will find it. But knowing how to split the other two
would be much harder without the labels: they run together. That is the challenge of
clustering in a nutshell.

The printed statistics above show the same thing: *setosa*'s petal length (about 1.5) is
nowhere near the other two (about 4.3 and 5.5), while *versicolor* and *virginica* have
means only about 1.2 apart with standard deviations around 0.5 — close enough that their
points intermingle.
:::

Clustering is useful for:

- **Data analysis** — discovering structure you didn't know was there
- **Dimensionality reduction**
- **Anomaly detection** — if we have well-defined clusters and then see points lying well
  outside them, that suggests anomalous samples worth analyzing further. In climate and
  environmental science this could be an unusual soil or water sample, an unusual weather
  pattern, or an instrument that has developed a fault and is reporting something not
  typically represented in the data set.

## There are many different clustering algorithms

Different algorithms are built on different principles, and behave very differently
depending on how your data is actually distributed. The notebook applies four common
methods to five differently-shaped data sets.

Rather than compare them by eye, we can score each clustering against the true grouping
with the **Adjusted Rand Index (ARI)**: 1.0 means the clustering exactly matches the truth,
and 0.0 means it is no better than random guessing.

```python
import time, warnings
from sklearn import cluster, mixture
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

n = 500
circles = make_circles(n_samples=n, factor=0.5, noise=0.05, random_state=0)
moons = make_moons(n_samples=n, noise=0.05, random_state=0)
blobs = make_blobs(n_samples=n, random_state=8)
Xa, ya = make_blobs(n_samples=n, random_state=170)
aniso = (Xa @ [[0.6, -0.6], [-0.4, 0.8]], ya)
varied = make_blobs(n_samples=n, cluster_std=[1.0, 2.5, 0.5], random_state=170)

datasets = [("circles", circles, 2), ("moons", moons, 2), ("varied", varied, 3),
            ("aniso", aniso, 3), ("blobs", blobs, 3)]

print(f"{'dataset':10s} {'K-Means':>10s} {'MiniBatchKM':>12s} {'DBSCAN':>10s} {'GaussMix':>10s}")
for dname, (Xd, ytrue), k in datasets:
    Xd = StandardScaler().fit_transform(Xd)
    models = [cluster.KMeans(n_clusters=k, n_init=10, random_state=0),
              cluster.MiniBatchKMeans(n_clusters=k, n_init=10, random_state=0),
              cluster.DBSCAN(eps=0.3),
              mixture.GaussianMixture(n_components=k, covariance_type="full", random_state=0)]
    scores = []
    for m in models:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xd)
        pred = m.predict(Xd) if hasattr(m, "predict") else m.labels_
        scores.append(adjusted_rand_score(ytrue, pred))
    print(f"{dname:10s} " + " ".join(f"{s:10.2f}" if i == 0 else f"{s:12.2f}" if i == 1
                                     else f"{s:10.2f}" for i, s in enumerate(scores)))
```

:::{admonition} What the plot shows
:class: note
A grid of 20 small scatter plots: five rows (one per data set) by four columns (one per
algorithm). Points are colored by the cluster each algorithm assigned. Each panel has its
runtime in milliseconds in the corner.

The rows are shaped as follows:

- **circles** — one ring of points inside another ring
- **moons** — two interlocking crescents
- **varied** — three round blobs whose spreads differ a lot (one tight, one very diffuse)
- **aniso** — three blobs stretched into long, tilted ellipses
- **blobs** — three round, well-separated, similar-sized blobs

Reading across, K-means colors the **circles** and **moons** rows by slicing them straight
down the middle — cutting *across* each ring or crescent rather than following it — which is
visibly wrong. DBSCAN follows the ring and crescent shapes correctly, but on **varied** it
merges or discards clusters. The Gaussian Mixture Model follows the tilted ellipses of
**aniso** that K-means cuts across. On **blobs**, everything works.

The ARI table above states this numerically: scores near 1.0 are the panels that look right,
near 0.0 the ones that look wrong.
:::

Reading across the rows, some methods do much better than others depending on the **shape**
of the clusters and how they sit relative to one another. K-means handles the well-separated
`blobs` beautifully and the `circles` and `moons` terribly. DBSCAN handles the non-convex
shapes but is thrown by `varied`, where cluster densities differ. The Gaussian Mixture Model
handles `aniso`, the stretched clusters, far better than K-means.

There is also a **runtime** trade-off. A method like **OPTICS** works well across many of
these data sets but is one of the most computationally expensive available. Depending on how
your data is distributed, how much time you have, and how large your data set is, some
methods will be more usable than others.

We'll cover K-means, DBSCAN, and Gaussian Mixture Models below, but there are many more. It
is worth exploring the [scikit-learn clustering
documentation](https://scikit-learn.org/stable/modules/clustering.html) for whatever data set
you are actually trying to cluster.

## K-means clustering

K-means is one of the classic, older clustering methods, and it has been around a very long
time.

The **assumption you must make as the user** is how many clusters are present in the data to
begin with. The number of clusters, $K$, is a **hyperparameter** — the algorithm cannot tell
you it.

Each sample $\mathbf{x}_i$ will be assigned one (and only one) cluster; samples cannot belong
to several clusters at once. We write $C(i)$ for the cluster number of the $i$th observation.
Our dissimilarity measure is the **Euclidean distance** — effectively the vector distance
between two points in feature space.

K-means minimizes the **within-cluster point scatter**: the sum, over every cluster $k$ and
every point assigned to it, of the squared distance from that point to the cluster's mean
$\mathbf{m}_k$.

$$W(C) = \sum_{k=1}^{K} \sum_{C(i)=k} \lVert \mathbf{x}_i - \mathbf{m}_k \rVert^2$$

### The algorithm

1. Randomly initialize $K$ initial centroids.
2. Assign each observation $\mathbf{x}_i$ to its nearest centroid.
3. For a given cluster assignment $C$, compute the cluster means — the mean of all points
   currently assigned to cluster $k$:

   $$\mathbf{m}_k = \frac{1}{N_k}\sum_{C(i)=k} \mathbf{x}_i$$

4. For the current set of cluster means, reassign each observation to its nearest centroid:

   $$C(i) = \arg\min_{1 \le k \le K} \lVert \mathbf{x}_i - \mathbf{m}_k \rVert^2$$

5. Iterate steps 3 and 4 until convergence — until the centroids stop moving.

```python
from sklearn.cluster import KMeans

# Five clusters, some more spread out than others.
X5, y5 = make_blobs(n_samples=800, centers=[[-2.5, 3], [0.5, 4.5], [-1.5, 0],
                                            [3, 1.5], [4, 3.5]],
                    cluster_std=[0.5, 0.4, 0.9, 0.35, 0.4], random_state=7)

rng = np.random.default_rng(2)
init = X5[rng.choice(len(X5), 5, replace=False)]   # random initial centroids

# Watch the centroids stop moving as the iterations proceed.
prev = init
print("iteration : total centroid movement : inertia")
for iters in range(1, 7):
    km = KMeans(n_clusters=5, init=init, n_init=1, max_iter=iters, random_state=0).fit(X5)
    move = np.linalg.norm(km.cluster_centers_ - prev, axis=1).sum()
    print(f"    {iters}     : {move:22.4f} : {km.inertia_:.1f}")
    prev = km.cluster_centers_
```

:::{admonition} What the plot shows
:class: note
Six panels showing the algorithm running. In each, the data is drawn as small black dots,
the five centroids as large red X markers, and the plane is shaded into five colored
regions — the **decision boundaries**, where every point in a shaded region is nearer to
that region's centroid than to any other.

**Panel 1** — five red X's dropped at random positions, some sitting in empty space or two
inside the same blob; no shading yet.
**Panel 2** — the same X's, now with the colored regions drawn, showing which points each
centroid has claimed. Some regions slice a real blob in half.
**Panels 3 to 5** — the X's visibly **march** toward the centers of the blobs, and the
boundaries between the colored regions shift with them.
**Panel 6** — converged: each red X sits in the middle of one blob, and each colored region
contains one blob.

The printed table above tracks the same process: the "total centroid movement" column shrinks
toward zero as the centroids settle, and the inertia falls and then levels off.
:::

Each update recomputes the centroid as the mean of the points currently assigned to it, which
pulls it toward the cluster's true center, which changes the assignments, and so on until
nothing moves. For a data set like this — well-separated, roughly round, comparable densities
— K-means does a good job.

### K-means does not have a unique solution

Because the centroids are **randomly initialized**, running K-means more than once will not
give exactly the same solution. Iteratively minimizing the **inertia** — scikit-learn's name
for exactly the $W(C)$ we wrote down above, the sum of squared distances from every point to
its assigned centroid — does not guarantee a unique answer. The algorithm can settle into
different local minima depending on where it started.

```python
for seed in [3, 0]:
    rng = np.random.default_rng(seed)
    init = X5[rng.choice(len(X5), 5, replace=False)]
    km = KMeans(n_clusters=5, init=init, n_init=1, random_state=0).fit(X5)
    print(f"random start {seed}: inertia = {km.inertia_:7.1f}, "
          f"agreement with true clusters (ARI) = {adjusted_rand_score(y5, km.labels_):.3f}")
```

This prints:

```
random start 3: inertia =   438.2, agreement with true clusters (ARI) = 0.988
random start 0: inertia =  1250.0, agreement with true clusters (ARI) = 0.718
```

:::{admonition} What the plot shows
:class: note
Two panels, each a converged K-means solution from a different random start, drawn the same
way as above (black dots, red X centroids, shaded regions).

They are **not the same**. In the poorer of the two, one centroid has settled *between* two
real blobs — splitting the difference and claiming half of each — while a second centroid has
been left to split one real blob into two colored halves. In the better solution each blob
gets exactly one centroid.

The printed inertia and ARI above tell you which is which without the picture, and are the
more precise version of the comparison: 438 against 1250, 0.99 against 0.72.
:::

Two different initializations, two genuinely different answers — and one is much worse than the
other. The good run finds the five real clusters almost exactly (ARI 0.99). The bad run gets
stuck in a **local minimum** with nearly three times the inertia, which is why it only scores
0.72.

Both are legitimate stopping points for the algorithm — neither can improve by moving a single
centroid. Nothing about the second run is a bug; it is just where that particular random start
happened to lead.

In practice `scikit-learn` guards against this by running the algorithm several times from
different random initializations and keeping the best result — that is what the `n_init`
parameter controls. Above we set `n_init=1` deliberately, to expose the underlying behavior.

## How to choose the number of clusters?

The other challenge is that we have to know $K$ *a priori*, and often we don't. If we are
looking at some environmental variables we have measured, we may have no idea how many classes
are actually represented in the data. Running K-means with $K=3$ or $K=8$ produces *an* answer
either way; neither tells you it is the wrong one.

### The elbow method

Plot the **inertia** — the sum of squared distances from every point to its assigned centroid —
against the number of clusters. Inertia always decreases as $K$ grows (with $K$ equal
to the number of samples it reaches zero), so we can't just minimize it. Instead we look for
the **elbow**: the point where the curve stops dropping steeply and starts decreasing much
more slowly.

```python
ks = list(range(1, 10))
inertias = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(X5).inertia_ for k in ks]

print("  K :  inertia : drop from K-1")
for i, (k, v) in enumerate(zip(ks, inertias)):
    drop = f"{inertias[i-1] - v:8.0f}" if i else "       -"
    print(f"  {k} : {v:8.0f} : {drop}")
```

:::{admonition} What the plot shows
:class: note
A single line falling steeply from left to right and then flattening: inertia (vertical)
against number of clusters K (horizontal, 1 to 9). It plunges from about 7500 at K=1 to about
440 at K=5, then runs almost flat from K=6 onward. An annotation with an arrow points at the
region around K=4 reading "somewhere around here?" — deliberately non-committal, because the
bend is gradual rather than sharp.

The printed table of drops above is the reliable way to read this curve.
:::

The inertia has an elbow, but **it is not always enough to objectively choose $K$**.

Look at the printed drops rather than the picture. Going from 4 to 5 clusters still buys a
substantial reduction in inertia (about 405), and only after $K=5$ does the curve really
flatten (about 77). So the bend is arguably at 5 — which is what we built. But the bend is
**gradual**, and that is the problem: reading this curve you could justify 4 or 5, and a
standard automated knee-finding rule applied to it actually returns **3**. The elbow gives you
a *range*, not an answer.

This is worse when clusters have different **densities**, as they do here: some are tightly
packed and some are spread out, and inertia is dominated by the spread-out ones, which keep
rewarding extra splits regardless of the true cluster count.

### The silhouette score

An alternative is the **silhouette score**, which balances the distance to a point's own
cluster against its distance to the nearest other cluster. For a single sample:

$$s = \frac{b - a}{\max(a, b)}$$

where

- $a$ = the **mean distance to the other instances in the same cluster** (how tight its own
  cluster is around it)
- $b$ = the **mean nearest-cluster distance** — the mean distance to the instances of the
  closest cluster that the sample is *not* part of

The score ranges from $-1$ to $+1$. Near $+1$ means the sample is comfortably inside its own
cluster and far from the next one; near $0$ means it sits on a boundary; negative means it has
probably been assigned to the wrong cluster. We average this over all samples.

```python
from sklearn.metrics import silhouette_score

ks = list(range(2, 10))
sil = [silhouette_score(X5, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X5))
       for k in ks]

print("silhouette score by K:")
for k, s in zip(ks, sil):
    marker = "  <-- peak" if s == max(sil) else ""
    print(f"  K = {k}: {s:.3f}{marker}")
```

:::{admonition} What the plot shows
:class: note
A single line that rises, peaks, and falls: silhouette score (vertical) against K
(horizontal, 2 to 9). It climbs from about 0.54 at K=2 to a clear maximum of about 0.68 at
**K=5**, marked with a dotted vertical line, then falls away to about 0.49 by K=9.

Unlike the inertia curve, this one has an actual peak you can point at — which is the whole
argument. The printed scores above give you the same maximum directly.
:::

The silhouette score peaks **unambiguously at 5** — the number of clusters we actually built
into the data — and by a clear margin over its neighbors (0.682 against 0.647 at $K=4$).
Where the inertia curve gave us a soft bend we had to squint at, the silhouette score gives us
a maximum we can just read off. That is what makes it the more objective of the two.

It is not infallible — it is still a heuristic, and it has its own bias toward convex,
well-separated clusters. But when the elbow is ambiguous, the silhouette score is usually the
better guide.

### Advantages and disadvantages of K-means

K-means is **fast and scalable**, and a very simple algorithm to use. But it can struggle with
clusters of **varying sizes**, **varying densities**, and **non-spherical shapes**.

Variants such as **accelerated K-means** and **mini-batch K-means** improve on efficiency,
speed, and scalability, but they don't change the underlying assumptions.

```python
# Ellipsoidal (anisotropic) clusters — K-means' classic failure case.
Xa_raw, ya_raw = make_blobs(n_samples=500, random_state=170)
X_aniso = Xa_raw @ [[0.6, -0.6], [-0.4, 0.8]]

km_a = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X_aniso)
print(f"K-means on stretched clusters: inertia = {km_a.inertia_:.0f}, "
      f"agreement with truth (ARI) = {adjusted_rand_score(ya_raw, km_a.labels_):.3f}")
```

This prints an inertia of about 577 and an **ARI of 0.555**.

:::{admonition} What the plot shows
:class: note
A scatter of points forming three long, narrow, **tilted ellipses** lying roughly parallel to
one another, like three cigars side by side. The points are colored by K-means' cluster
assignment, with three red X centroids.

The coloring **cuts across** the ellipses rather than following them: each colored group is
a roughly circular chunk containing parts of more than one cigar, and the centroids sit
between the true clusters rather than along them.
:::

These clusters are stretched and skewed rather than round. K-means slices across them, placing
centroids *between* the true clusters, because it is minimizing squared distance to a center
and has no concept of a cluster having a *direction*.

Note what the ARI is telling us. K-means converged happily and reported a perfectly respectable
inertia — by its own objective, nothing went wrong. But scored against the clusters that are
actually there, it only manages about 0.56. **A low inertia does not mean a good clustering**;
it only means the algorithm did well at the thing it was optimizing, which is not the same
thing.

K-means works best when clusters have **similar densities**, are **reasonably well separated**,
and are **roughly spherical** — the `blobs` case in the comparison table above. Once you have
more complex shapes or differing densities, you need something else.

## DBSCAN

**DBSCAN** — Density-Based Spatial Clustering of Applications with Noise — takes a different
approach. It finds **core samples of high density** and then expands clusters outward from
them. Where K-means is top-down (start with $K$ centers, assign everything), DBSCAN is
**bottom-up**: find the dense regions of the data set, then grow them by adding neighboring
points.

DBSCAN works well when the data contains clusters of **similar density**, separated by regions
of **lower density**.

Its key hyperparameter is **`eps`**: the maximum distance between two samples for one to be
considered a **neighbor** of the other. This is the most important parameter to get right for
DBSCAN, and it makes a large difference.

```python
from sklearn.cluster import DBSCAN

X_moons, y_moons = make_moons(n_samples=1000, noise=0.06, random_state=42)

km_m = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X_moons)
print(f"K-means (K=2)   : ARI = {adjusted_rand_score(y_moons, km_m.labels_):.3f}")

for eps in [0.05, 0.2]:
    db = DBSCAN(eps=eps).fit(X_moons)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = (db.labels_ == -1).sum()
    print(f"DBSCAN eps={eps:<4}: ARI = {adjusted_rand_score(y_moons, db.labels_):.3f}, "
          f"found {n_clusters} clusters, {n_noise} points marked noise")
```

:::{admonition} What the plot shows
:class: note
Three panels, all of the same data: two interlocking crescent "moons", one curving up and one
curving down, nested into each other.

**Left — K-means (K=2)**: the coloring splits the data with a straight boundary running
through the middle, so each colored group contains half of *each* moon. A straight decision
boundary cannot wrap around a crescent.

**Middle — DBSCAN, eps = 0.05**: the moons are shattered into **many** small differently
colored fragments, each a few points long, plus scattered noise points. The value is too low,
so no region qualifies as densely connected.

**Right — DBSCAN, eps = 0.2**: exactly two colors, each tracing one complete crescent from
end to end. Correct.

The printed ARI and cluster counts above capture all three outcomes numerically.
:::

In contrast to K-means, DBSCAN does a good job of capturing these **u-shaped** clusters, which
K-means simply cannot.

But look at the effect of `eps`. Set too low, DBSCAN tries to break each moon into many
separate sub-clusters, finding far more clusters than are really there. Set appropriately, it
identifies the two real clusters cleanly. Setting these hyperparameters correctly makes a big
difference to how well the algorithm performs.

DBSCAN is also **robust to outliers**, more so than K-means. It is allowed to label a point as
belonging to *no* cluster at all — those are the "noise" points counted above. K-means, by
contrast, always uses every point in the data set, because it recalculates centroids as means;
a few distant outliers can drag a centroid away and throw off the whole clustering. DBSCAN just
excludes a point that is too far from any cluster and where the density is too low. If your
data set has outliers, DBSCAN may be preferable.

## Gaussian Mixture Models

A **Gaussian Mixture Model (GMM)** is a **probabilistic** model that assumes all data points
are generated from a mixture of a finite number of Gaussian distributions with unknown
parameters.

You can think of a GMM as a **generalization of K-means** that also includes information about
the **covariance structure** of the data. K-means only knows where a cluster's center is. A GMM
also knows its *shape* and *orientation* — so a cluster that has been stretched and skewed
relative to a typical Gaussian is still something it can represent.

In practice, that means a GMM handles the ellipsoidal clusters that defeated K-means.

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42).fit(X_aniso)
labels_gmm = gmm.predict(X_aniso)

print("On the same stretched clusters that defeated K-means:")
print(f"  K-means         : ARI = {adjusted_rand_score(ya_raw, km_a.labels_):.3f}")
print(f"  Gaussian Mixture: ARI = {adjusted_rand_score(ya_raw, labels_gmm):.3f}")
```

:::{admonition} What the plot shows
:class: note
Two panels of the same three tilted, cigar-shaped clusters.

**Left — K-means**: as before, the colors cut *across* the ellipses, and the red X centroids
sit between the true clusters.

**Right — Gaussian Mixture Model**: each colored group now follows one complete ellipse along
its full length, and the shaded decision regions are themselves stretched and tilted to match
the data rather than being round. The centroids sit along the ellipses.

The two ARI scores printed above state the difference plainly: the GMM's is far closer to 1.
:::

This prints an ARI of **0.555** for K-means against **1.000** for the Gaussian Mixture Model.

This is the same data set that K-means struggled with. Where K-means put its centroids between
the ellipsoidal clusters and scored about 0.56, the GMM recovers the true clusters **exactly** —
because it can model the covariance, the stretch and tilt, of each one.

A GMM is also *probabilistic*: rather than a hard assignment, `predict_proba` gives you the
probability that each point belongs to each cluster, which is useful when you want to express
uncertainty about boundary cases. (On this particular data set the clusters are well enough
separated that almost every point is assigned with near-total confidence, so there is little
uncertainty to look at — but the machinery is there when your clusters do overlap.)

As with everything here, it depends on your specific data: how the points are laid out in your
clusters, and what shape those clusters are.

## Summary

| Method | Key hyperparameter | Good at | Struggles with |
|---|---|---|---|
| **K-means** | `n_clusters` | Fast, scalable, round well-separated clusters of similar density | Varying sizes/densities, non-spherical shapes, outliers; no unique solution |
| **DBSCAN** | `eps` | Non-convex shapes; robust to outliers; finds its own cluster count | Clusters of differing density; sensitive to `eps` |
| **Gaussian Mixture** | `n_components` | Ellipsoidal clusters; covariance structure; probabilistic assignments | Still assumes Gaussian components; need to choose the component count |

There are a number of different clustering methods beyond these three, built on different
principles and suited to different data. Explore the [scikit-learn
documentation](https://scikit-learn.org/stable/modules/clustering.html) for the data set you
are working with.

Recall too that clustering is often used **in combination with** [dimensionality
reduction](dimensionalityreduction.md): first find a transformation into a space where the
classes separate more easily, then cluster in that space. Together they give us **unsupervised
classification**, without ever starting from labeled data.