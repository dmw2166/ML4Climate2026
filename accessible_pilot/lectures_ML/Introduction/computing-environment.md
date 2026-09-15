# Computing Environment for Class

:::{admonition} Accessible workflow — read this first
:class: important
The steps below describe the **standard** browser-based JupyterHub interface. If
you use a screen reader, use the **accessible workflow** instead — VS Code with
Python scripts and a readable terminal, connecting to the same class hub. It is
set up once and reused for the whole course:

**[Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html)**
(from CLMT5045; applies unchanged here). The datasets, commands, and Python are
identical — only the interface changes.
:::

## Our Course JupyterHub

[JupyterHub](https://jupyter.org/hub) is a multi-user Jupyter environment
designed for companies, classrooms and research labs. We have a JupyterHub for
the course in which you should do all of your assignments and take your notes.
You should be receiving an email from Dr. Gus Correa, Systems Programmer and
Administrator at LDEO (Prof. Westervelt's home institution within Columbia
University). The physical server is called "chopin" and lives at LDEO.

You log in to the JupyterHub running on chopin, with the username and password
provided to you by email, at
<https://chopin.ldeo.columbia.edu:8441/jupyterhub/>.

Everyone is given a home directory with a quota of 10 GB, and a work directory
at `/data9/G5120/work/your_username` with a quota of 50 GB. Code can go in the
home directory, which is backed up, but large datasets should go in the work
directory.

Please remember to actively log out of the JupyterHub when you are done using
it. It is not enough to simply close the browser or close your laptop screen.

You should already be familiar with JupyterHub from CLMT 5045. If you need a
refresher, please check out the material
[here](https://earth-ds-ml.github.io/summer_2026/lectures_DS/computing_env/jupyterlab_and_colab.html),
or, for the screen-reader version, the [accessible setup
guide](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).

## Computing environment and software libraries for machine learning

There are a number of different software libraries that can be used for machine
learning. Python is the most popular programming language used for machine
learning. Many of the most popular libraries for machine learning have been
developed in Python, including `PyTorch`, `TensorFlow`, `JAX`, and `Keras`.
Julia, a newer programming language focused on performance computing in
scientific and technical fields also supports machine learning libraries.

In this class, we will work with Python. We will use classic machine learning
algorithms in the `sci-kit learn` library, as well as deep learning models
implemented in `Tensorflow`.