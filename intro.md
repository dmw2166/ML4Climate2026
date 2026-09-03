# Welcome to Machine Learning and Artificial Intelligence for Climate Science and Solutions
Fall 2026 Course Website. This is a 3-credit class in the Columbia Climate School taught by [Prof. Dan Westervelt](https://aerosol.ldeo.columbia.edu/). This course website was developed from an [original version](https://kdlamb.github.io/ML4Climate2026/intro.html) created by [Dr. Kara Lamb](https://kdlamb.github.io/). 


## Course Description
Machine learning (ML) and artificial intelligence (AI) tools are quickly becoming part of the standard toolkit in climate research and problem solving. From improving climate model predictions to tracking emissions and mapping land use change with satellite data, ML methods are helping scientists make sense of enormous, complex datasets and evaluate potential climate solutions. As these tools spread across the field, research scientists and policymakers alike need a working understanding of what ML and AI can and cannot do.

This course is a hands-on introduction to machine learning and AI with applications in climate science. Students will learn the fundamentals of building, training, and evaluating ML models, including data preprocessing, model selection, and performance metrics. Working with real climate and environmental datasets, students will gain practical coding experience in Python and a realistic sense of both the power and the limits of ML in climate research. 


## Course objectives
By the end of this course, students will:
- Develop a strong foundation in ML/AI concepts and techniques applicable to climate science and problem solving
- Apply ML methods to analyze observational data sets, predict trends in environmental data, and identify patterns.
- Critically assess the strengths and limitations of ML approaches to climate and environmental applications.
- Become familiar with climate model emulation, atmospheric foundation models, and ML/AI weather forecasting.
- Apply ML to remote sensing and geospatial data analysis, including land cover classification, plume detection, and other examples.
- Apply ML to model evaluation, extrapolation, uncertainty quantification, and interpretability (explainable AI)
- Understand the principles of physics-informed ML. 
- Understand ethical considerations and policy implications related to ML/AI applications to climate research and solutions.


## Course structure

We meet **Mondays 4:10 - 6:40** at the Forum. Most classes will include at least 2 of the 3 elements: a lecture covering new material, a hands-on coding tutorial guided by the instructor, and/or collaborative time with classmates and the instructor to work on class assignments. Bring a laptop to every class as you'll spend a large fraction of class time writing and running code, not just listening. We use the class server Chopin and its JupyterHub (further details [here](https://dmw2166.github.io/ML4Climate2026/lectures_ML/Introduction/computing-environment.html)) to access datasets and to train and evaluate models in Jupyter notebooks.

## Assignments and grading

Your grade in this course is based on:

- **50%** — coding assignments
- **40%** — final project and presentation
- **10%** — attendance and participation


**Coding assignments.** There are **ten** coding assignments, assigned each week, following the material covered in class. Each is due by **Monday at midnight of the following week** (e.g. the assignment given at the start of week 2 is due Monday of week 3). 


## Final project

The final project counts for **40%** of your grade, in lieu of a final exam. You will find a
real environmental or climate dataset, pose a question you can answer with it, and apply what
you have learned to answer that question.


### Choosing a dataset

Pick something real. It should be observational, model, or remote sensing data with genuine
environmental relevance and not a cleaned teaching dataset with a known answer.

**You are encouraged to use data from your own research.** If you have a dataset you already
care about, this is a good excuse to try these methods on it.

Some starting points: ERA5 reanalysis (Copernicus), CMIP6 output (ESGF or the LEAP catalogs),
NOAA GHCN station records and OISST, EPA AQS or OpenAQ for air quality, Sentinel-2 and Landsat
imagery via the Planetary Computer, CAMELS for hydrology, ClimateBench for emulation.

Bring your dataset choice to office hours early if you are unsure whether it will work.

### What your analysis must include

1. **A clear question.** State what you are predicting or discovering, and why it matters.
   "I applied a random forest to this dataset" is not a question.
2. **A baseline.** Compare against the appropriate naive predictor — climatology, persistence,
   the majority class, or simple linear regression. Report it alongside your model.
3. **A validation scheme that matches the application.** If your data has spatial or temporal
   structure, a random train/test split will overstate your skill. Justify the scheme you
   chose. This is the single most heavily weighted technical element.
4. **Some characterization of uncertainty.** Error bars, an ensemble spread, or a clear
   statement of what you could not quantify and why.
5. **Interpretation.** What did the model learn? Does it agree with physical intuition? Where
   does it fail, and is that failure telling you something?
6. **Limitations.** What would you need to trust this result — more data, different data, a
   different method? What are the practical or ethical considerations if someone acted on it?

```{admonition} Negative results are fine
:class: tip
A negative result, honestly obtained and clearly explained, receives full credit. If your model
cannot beat persistence, that is a finding. Report it.
```

### Deliverables

- **Code** — a GitHub repository with your notebook(s) and a README that lets someone reproduce
  your work
- **Paper** — 5 pages, excluding figures and references
- **Presentation** — 10 minutes plus questions, in class on December 14

### Timeline

| Milestone | Date |
| --- | --- |
| Topic proposal (one paragraph: dataset, question, method) | Monday, November 9 |
| In-class presentations | Monday, December 14 |
| Paper and code due | Monday, December 21 |

### Grading

| Component | Weight |
| --- | --- |
| Question and dataset — well-posed, appropriate, genuinely used | 15% |
| Methods — sound choices, correctly implemented | 25% |
| Validation and baselines — honest, justified, correctly applied | 25% |
| Interpretation and limitations — critical, physically grounded | 20% |
| Communication — paper and presentation | 15% |


## Using AI in This Course

Large Language Models (LLMs) such as ChatGPT, Google Gemini, Claude, etc., are rapidly changing norms in higher education. Some of what these tools provide can be useful and helpful, but overreliance on LLMs is a risk. Asking an LLM to do an assignment for you will do you a major disservice in your future career, as you will not be learning and acquiring the skills needed to succeed in a career in climate. Additionally, the LLMs are prone to mistakes. Use of LLMs is allowed in this class; however, you must cite your use of it in your assignments. You may use whichever one you like, but note that Google Gemini is available for free to Columbia students: [https://www.cuit.columbia.edu/content/google-gemini]. An example of acceptable use citation might look like “Used Claude Fable 5 to look up syntax for creating a filled contour plot”. Unacceptable use would be copying and pasting code that you can’t explain. You are responsible for understanding and being able to explain your work.  The instructor and/or TAs reserve the right to ask you questions about your assignment if they suspect over-reliance on LLMs. If you are not understanding the code or material, that is a sign to revisit the material, ask questions, go to office hours, etc., and not to rely further on LLMs. 

**How to use it — Socratic mode.** Default to asking the AI to *teach* you, not to *do it for you*. At the start of a working session, prime your chat with a tutor prompt. Here is an example. Feel free to adapt it as you learn what works:

```
You are acting as my Socratic Tutor for a graduate-level Machine Learning and Climate Science
course. I am going to show you bugs or ask about Python/Xarray/Git.

Rules:
1. NEVER give me corrected code blocks or direct syntax fixes.
2. Explain the computational or data concept I am missing.
3. Ask me ONE targeted guiding question to help me find the solution myself.
```

The goal is to use AI to build understanding, not to paste solutions you can't explain.

**Important! Do not install any AI tool coding companion (e.g. Claude Code) directly onto our course server, chopin!** Not only is this over-reliance on AI, it will degrade the computing experience for others in the class by hogging memory and CPU time.

**What AI is good at, and what it isn't.** Chat-based AI is genuinely useful for explaining error messages, suggesting matplotlib syntax, walking through an unfamiliar library API, or summarizing what a function does. It is less reliable for judging whether your scientific result is correct, picking the right analysis for *your* data, catching subtle bugs in numerical or coordinate-system code, or knowing what "looks right" for a specific geophysical field. Treat AI as a fast, broadly-read but inexperienced collaborator — useful for the syntax layer, not a substitute for your own scientific judgment.


