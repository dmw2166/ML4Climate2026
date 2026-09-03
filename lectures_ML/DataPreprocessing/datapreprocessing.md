# Data Exploration and Data Pre-Processing

Data exploration and pre-processing are the first steps in any machine learning pipeline. Data exploration refers to visualizing and understanding
our data sets before we start applying any machine learning algorithm. Data-preprocessing is the steps we perform to prepare and standardize data
sets to be fed into machine learning algorithms.

## Data Exploration
An important step in applying machine learning methods is data exploration. This typically consists of exploring the distributions of the different 
variables in our data sets before we start trying out different machine learning models.

We want to ask questions like:
- Are the variables normally or log-normally distributed?
- Are the distributions highly skewed?
- Are some of the variables strongly correlated with one another?
- Do we have missing values or outliers in our data sets?
- If the variables are categorical (i.e. divided up into different classes), how many different classes are represented in our data sets?
- Are the classes relatively balanced (i.e. do we have a similar number of samples for each class?), or are they imbalanced (do some classes have
a lot of samples and some have very few?)
- What order of magnitude are the variables in our data sets?
- Are variables strongly spatially or temporally correlated?

All of these are important questions to understand because they can affect the performance of our machine learning algorithms and lead to biased or
unexpected results when we apply algorithms. Taking time to first explore and understand these types of questions can inform what methods we will 
use to preprocess the data sets, which algorithms might be the most appropriate for our problem, and save time later.

## Data Pre-processing
Typically, we will devote at least 80% of the time we spend on a machine learning project to preparing and managing the data sets. Once the 
data set is prepared, we can then try out (or benchmark) a number of different algorithms. Data preprocessing refers to analyzing, filtering, transforming,
and encoding data such that it is ready to pass to the machine learning algorithm.

Next, we will work through a tutorial on data preprocessing. This will use the class GitHub repo. If you need reminders about how to access and download
files to LEAP JupyterHub or Google Colab, you can refresh yourself here:
[Intro to Git](https://dmw2166.github.io/comp-res-climate-data-sci/lectures/environment/intro_to_git.html).

