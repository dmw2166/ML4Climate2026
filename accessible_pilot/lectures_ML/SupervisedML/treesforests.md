# Decision Trees and Random Forests

:::{admonition} How to work through these notes (accessible version)
:class: important
On the standard site this page is a Jupyter notebook. Here, the code is written to be run as a
**Python script** in VS Code — see [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Put the code into a file such as `treesforests.py`, adding the sections below one at a time, and
run it with `python treesforests.py`. Read printed output with the terminal's Accessible View.

**Plots.** Each figure has a **"What the plot shows"** description, and the code prints the
numbers behind it. To explore a plot yourself with sound and a text/braille readout, use
**MAIDR** — see [Accessible plots with
MAIDR](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/sci_python/trying_maidr.html).

**The tree diagram.** The standard page draws the trained tree as a flowchart image. That
picture is not the only way to read a tree — scikit-learn's `export_text` prints the **same
tree** as an indented text outline, which a screen reader handles far better than any diagram.
We use it below, and it is genuinely the better tool here, not a consolation prize.
:::

This page covers two closely related supervised models: the interpretable **decision tree**, and
the **random forest** ensemble that combines many trees for accuracy and robustness. Both handle
classification *and* regression.

## Growing a decision tree

A decision tree recursively **partitions the feature space**. At each node it picks a feature and
a threshold, sending samples left or right depending on whether the feature is above or below it.
The goal at every split is to make the resulting groups as **pure** as possible — ideally each
leaf contains one class.

Purity is measured by the **Gini impurity** of a node with class fractions $p_k$:

$$G = 1 - \sum_{k} p_k^2$$

$G = 0$ is a perfectly pure node. The **CART** (Classification And Regression Trees) algorithm
greedily chooses the split that most reduces impurity — for regression it minimizes MSE within
each group instead.

```python
import numpy as np

# Gini impurity for a two-class node as the class balance varies.
print("  fraction in class 1 : Gini impurity")
for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    print(f"           {p:4.2f}       : {1 - (p**2 + (1-p)**2):.3f}")
```

:::{admonition} What the plot shows
:class: note
A single smooth curve, an upside-down arch (a downward-opening parabola). Gini impurity is on the
vertical axis, the fraction of samples in class 1 on the horizontal axis, from 0 to 1.

The curve starts at **0** on the left (a node containing only class 0 — perfectly pure), rises to
a maximum of **0.5** at the center where the classes are split 50/50 (maximally impure — the node
tells you nothing), then falls symmetrically back to **0** at the right (only class 1 — pure
again).

The printed table is the same curve as numbers. The shape is the message: impurity is worst when
a node is evenly mixed, and a split is "good" exactly to the extent that it moves its children
*away* from the center of this arch and toward either end.
:::

### Training and reading a tree

Because every split is a simple threshold, a trained tree is a flowchart you can read top to
bottom — this transparency is a big part of why trees are popular in science. Here is a shallow
tree on the Iris data set.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

iris = load_iris()
X, y = iris.data, iris.target
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

# The whole tree, as text. This is the accessible equivalent of plot_tree.
print(export_text(tree, feature_names=list(iris.feature_names)))
print("class 0 = setosa, 1 = versicolor, 2 = virginica")
print(f"training accuracy: {tree.score(X, y):.3f}")
```

This prints:

```
|--- petal width (cm) <= 0.80
|   |--- class: 0
|--- petal width (cm) >  0.80
|   |--- petal width (cm) <= 1.75
|   |   |--- petal length (cm) <= 4.95
|   |   |   |--- class: 1
|   |   |--- petal length (cm) >  4.95
|   |   |   |--- class: 2
|   |--- petal width (cm) >  1.75
|   |   |--- petal length (cm) <= 4.85
|   |   |   |--- class: 2
|   |   |--- petal length (cm) >  4.85
|   |   |   |--- class: 2
```

:::{admonition} What the plot shows
:class: note
The standard page draws this as a flowchart: rounded boxes connected by lines, fanning downward
from a single root box at the top through three levels. Each internal box states its split rule
(e.g. "petal width (cm) <= 0.8"), its Gini impurity, the number of samples reaching it, and the
class counts; each box is color-shaded by its majority class, with stronger color meaning purer.
Leaf boxes at the bottom have no split rule.

The text outline above contains the **same information, in the same structure** — indentation is
the tree depth, and each `|---` line is a branch. You are not missing anything by reading it this
way.
:::

Read the outline as nested if/then rules. The whole model is four decisions:

- **Petal width ≤ 0.8** → **setosa**, immediately. One threshold on one feature isolates that
  species completely — which matches what we saw in [classification](classification.md), where
  setosa sat in its own corner far from the others.
- Otherwise, **petal width ≤ 1.75** → probably **versicolor**, refined by petal length at 4.95.
- Otherwise → **virginica**.

Notice the tree uses only **petal** measurements. It never splits on sepal length or width at all
— it found those uninformative and silently dropped them. That is the "implicit feature selection"
listed among the advantages below, visible in a concrete model.

:::{admonition} A quirk worth spotting
:class: tip
Look at the last two lines: the split on `petal length <= 4.85` sends samples to **class 2 either
way**. The split changes nothing.

This is an artifact of forcing `max_depth=3`. That node was already nearly pure, so CART made the
best split available even though it cannot alter a single prediction. It is harmless here, but it
is a good reminder that a tree's structure is not automatically meaningful — some branches are
just the algorithm filling the depth it was given.
:::

The splits carve the feature space into **axis-aligned rectangles**. Plotting the decision regions
for two features makes the boxy boundaries obvious — quite different from the smooth linear
boundary of logistic regression.

```python
X2 = iris.data[:, 2:4]      # petal length, petal width
t2 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X2, y)
print(export_text(t2, feature_names=["petal length", "petal width"]))
```

:::{admonition} What the plot shows
:class: note
A scatter of the 150 iris flowers (petal width vertical, petal length horizontal) over a
background divided into colored regions — one per class.

The crucial visual detail is the **shape of the boundaries**: they are made entirely of
**horizontal and vertical straight segments meeting at right angles**, like a brick wall or a
city block plan. There are no slanted or curved edges anywhere.

This follows directly from the text outline above. Every split is of the form "feature ≤
threshold", which is geometrically a line perpendicular to that feature's axis. A tree can only
ever cut the plane with horizontal and vertical lines, so its regions are always rectangles.
Compare this with the single slanted straight line of logistic regression, or the smoothly graded
probability surface — a tree's boundary is blocky by construction.
:::

### Regularization and overfitting

Left unchecked, a tree keeps splitting until every leaf is pure — memorizing the training data.
Key hyperparameters control complexity: `max_depth`, `min_samples_split`, `min_samples_leaf`,
`max_leaf_nodes`, `max_features`. **Pruning** can also remove low-value branches after training.

The plot below shows train vs. test accuracy as the tree deepens.

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

Xm, ym = make_moons(n_samples=400, noise=0.3, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=0.3, random_state=0)

print(" depth : train : test  : gap")
for d in range(1, 16):
    m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
    a, b = m.score(Xtr, ytr), m.score(Xte, yte)
    print(f"   {d:2d}  : {a:.3f} : {b:.3f} : {a-b:.3f}")
```

:::{admonition} What the plot shows
:class: note
Two lines against `max_depth` (horizontal, 1 to 15), accuracy on the vertical axis.

The **train** line (circles) rises steadily and relentlessly from about 0.81 at depth 1 to a flat
**1.000** from depth 11 onward — a perfect score on the data it was fitted to.

The **test** line (squares) rises sharply to about 0.91 by depth 2, and then **stops improving**.
It wobbles in a narrow band between 0.88 and 0.91 for the rest of the plot and settles at 0.883.

The two lines start almost on top of each other and steadily **diverge**. That widening gap is the
overfitting signature.
:::

Read the printed `gap` column — it is the clearest signal here. At depth 1 the gap is essentially
zero (0.814 train vs 0.817 test — the model is equally mediocre on both). By depth 15 the gap is
**0.117**: a perfect 1.000 on training data and 0.883 on data it has not seen. Everything the tree
learned after depth ~2 was **noise in the training set**, not signal it could carry to new data.

:::{admonition} Keep the size of the effect in proportion
:class: warning
It is tempting to describe this as "test accuracy peaks and then collapses". It does not. Test
accuracy peaks at **depth 2** (0.908) and ends at **0.883** — a sag of only about 2.5 percentage
points, and it wobbles up and down on the way (it touches 0.908 again at depth 7). With only 120
test points, those wiggles are mostly noise.

The dramatic part of overfitting here is not that test performance falls off a cliff — it is that
**train performance keeps climbing to a perfect score while test performance flatlines**. Nine
extra levels of tree bought you a great deal of apparent accuracy and no real accuracy at all.
:::

### Advantages and limitations of decision trees

**Advantages**

- Simple to understand; every decision path can be traced and explained.
- Implicitly performs feature selection (ignores redundant variables).
- Handles numerical *and* categorical features; robust to outliers.

**Limitations**

- Prone to **overfitting** and **high variance** — small data changes can reshape the tree.
- Large trees become hard to interpret.
- Inference is fast, but *training* is the expensive part: growing the tree scales roughly as
  $O(m \cdot N \log N)$.

## Random forests

A **random forest** is an *ensemble* of decision trees. Each tree is trained on a random subset of
the data (and a random subset of features at each split), and their predictions are combined.
Diversity among the trees is the key: if the trees make *different* errors, averaging them cancels
the noise, reducing variance and overfitting.

**How the samples are drawn:**

- **Bagging** = bootstrap sampling *with* replacement.
- **Pasting** = sampling *without* replacement.

**How the votes are combined:**

- **Hard voting** — each tree votes for one class; the majority wins.
- **Soft voting** — average the predicted class probabilities, then pick the highest. Usually
  preferred because it accounts for each tree's confidence.

The forest's decision boundary is noticeably **smoother** than a single tree's.

```python
from sklearn.ensemble import RandomForestClassifier

single = DecisionTreeClassifier(max_depth=None, random_state=0).fit(Xm, ym)
forest = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xm, ym)

xx, yy = np.meshgrid(np.linspace(Xm[:, 0].min()-0.5, Xm[:, 0].max()+0.5, 300),
                     np.linspace(Xm[:, 1].min()-0.5, Xm[:, 1].max()+0.5, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

# "Smoother" made measurable: count how many adjacent grid cells fall on
# opposite sides of the boundary. Fewer = a shorter, less jagged boundary.
print("model             : boundary cells : train acc")
for name, m in [("single tree     ", single), ("forest (200)    ", forest)]:
    Z = m.predict(grid).reshape(xx.shape)
    edges = (np.diff(Z, axis=0) != 0).sum() + (np.diff(Z, axis=1) != 0).sum()
    print(f"{name}  : {edges:14d} : {m.score(Xm, ym):.3f}")

# Both memorize the training data perfectly. Do they generalize equally well?
print("\nOn a held-out split:")
for name, M in [("single tree ", DecisionTreeClassifier(random_state=0)),
                ("forest (200)", RandomForestClassifier(n_estimators=200, random_state=0))]:
    m = M.fit(Xtr, ytr)
    print(f"  {name}: train {m.score(Xtr, ytr):.3f} | test {m.score(Xte, yte):.3f}")
```

:::{admonition} What the plot shows
:class: note
Two panels of the same 400 "moons" points (two interlocking crescents with heavy noise, so the
two classes overlap through the middle), each shaded into two colored decision regions.

**Left — "single decision tree"**: the boundary is aggressively **blocky and staircase-like**, all
right angles. It is riddled with thin rectangular slivers and, crucially, **three separate visible
stray blocks** of one class stranded inside the other class's territory — each one a rectangle
drawn around a handful of noisy training points that the tree bent over backward to capture.

**Right — "random forest (200 trees)"**: the boundary follows roughly the same path but is far
**smoother and more coherent** — still slightly stepped (it is still made of trees), but the
staircase is fine-grained enough to read as a curve. The stray islands are essentially gone: one
clean region per class.

The printed counts measure exactly this. The single tree's boundary spans **2327** adjacent grid
cells; the forest's spans **1239** — a boundary a little over **half as long** through the same
data. Averaging 200 trees did not change the *kind* of boundary a tree can draw; it cancelled the
idiosyncratic jags each individual tree invented.
:::

Both models score a perfect **1.000** on the training data — by that number they are
indistinguishable, and both look like they have "learned" the moons. On held-out data the single
tree scores **0.883** and the forest **0.917**. The forest converts those cancelled jags into
about 3.4 points of real accuracy, on data neither model had seen.

### Why (and why not) use a random forest?

**Pros** — usually more accurate than a single tree; more representation power; smoother, more
robust predictions; provides **probabilistic** predictions and **uncertainty quantification**
through the spread across trees.

**Cons** — computationally more expensive (scales with the number of trees); less interpretable
than one tree; can increase bias when each tree sees only a small subset.

That "uncertainty quantification" is worth making concrete — it is just the vote spread:

```python
f = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
probs = f.predict_proba(Xte)[:, 1]     # fraction of the 200 trees voting class 1

print(f"test points where the trees nearly agree (>90% or <10%): {((probs>0.9)|(probs<0.1)).sum()} of {len(probs)}")
print(f"test points where the trees are split (40-60%):          {((probs>=0.4)&(probs<=0.6)).sum()} of {len(probs)}")
```

Most points (86 of 120) get a near-unanimous vote — the forest is confident. A handful (3 of 120)
land near a coin flip, and those sit in the overlap zone where the two moons interleave. A single
tree cannot tell you this: it outputs one class with no notion of how close the call was. The
forest gets it for free from the disagreement among its trees.

## Feature importance

Random forests can report **feature importance** — how much each feature reduces Gini impurity on
average across all trees and splits. This is a route toward *interpretable* models: it tells you
which variables actually drive the predictions.

In environmental science this helps decide which (possibly expensive to measure) features to
collect, which pollutants or drivers matter most, and how to trim a model down to its most
informative inputs.

:::{note}
Impurity-based importance is fast but biased toward high-cardinality and continuous features. When
importances drive scientific conclusions, cross-check with **permutation importance**
(`sklearn.inspection.permutation_importance`), which measures how much the score drops when a
single feature is shuffled.
:::

```python
import pandas as pd

rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(iris.data, iris.target)
imp = pd.Series(rf.feature_importances_, index=iris.feature_names).sort_values()
print(imp.round(3))
```

:::{admonition} What the plot shows
:class: note
A horizontal bar chart with one bar per feature, sorted shortest at the bottom to longest at the
top, measuring "importance (mean decrease in Gini impurity)".

Two bars are long and nearly equal — **petal length (0.438)** and **petal width (0.434)** — and
two are short: **sepal length (0.100)** and a barely-visible **sepal width (0.028)**.

The printed values are the bar lengths, so you lose nothing by reading them instead:

```
sepal width (cm)     0.028
sepal length (cm)    0.100
petal width (cm)     0.434
petal length (cm)    0.438
```
:::

The two petal measurements together account for about **87%** of the total importance, and sepal
width for under 3%. This is the same conclusion the single tree reached structurally by never
splitting on the sepal features at all — but now as a graded ranking rather than a yes/no. If
measuring sepal width were expensive, this chart is your evidence for dropping it.

## Beyond bagging: boosting and stacking

Random forests build trees **in parallel**. Two other ensemble ideas came up in lecture:

- **Boosting** — train weak learners **sequentially**, each correcting its predecessor. *AdaBoost*
  up-weights misclassified samples; *gradient boosting* fits each new tree to the residuals of the
  previous ones.
- **Stacking / blending** — train a final "meta" model to combine the predictions of several
  different base models, learning *when* to trust each one.

## Summary

This page covered **decision trees** and the **ensemble methods** built on top of them — a family
of models that drop the straight-line assumption and instead partition the feature space into
regions.

- A **decision tree** repeatedly splits the data on one feature at a time, choosing the split that
  most reduces **Gini impurity** (the CART algorithm). The result is a set of axis-aligned regions,
  each assigned a class (or a mean, for regression).
- Trees are **highly interpretable** — you can read the decision path as a sequence of if/then
  rules — and need little data preprocessing (no scaling required). But a single deep tree
  **overfits** and has **high variance**: small changes in the data can produce a very different
  tree.
- **Random forests** tame that variance by averaging **many diverse trees**. Two sources of
  diversity make this work:
  - **Bagging** — each tree is trained on a bootstrap sample (random draw with replacement) of the
    training data.
  - **Random feature subsets** — at each split, only a random subset of features is considered.

  Averaging (or majority **voting** for classification) over these decorrelated trees gives
  smoother, more accurate, and probabilistic predictions than any single tree.
- **Feature importance** from a forest ranks which variables drive the predictions — a practical
  tool for interpretable environmental modeling. Keep in mind that the default impurity-based
  importance is biased toward high-cardinality features; **permutation importance** is a more
  reliable alternative.
- Beyond bagging, **boosting** (train weak trees sequentially, each correcting its predecessor) and
  **stacking** (learn a meta-model to combine several base models) are two other ways to build
  strong ensembles from simple trees.

Next: [Tutorial on Random Forests (Wildfire cause prediction)](randomforests.md), which
applies a random forest end-to-end to a real environmental data set.
