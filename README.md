isea
==============================

A short description of the project.

Project Organization
---

    .
    ├── LICENSE
    ├── README.md   <- The top-level README for developers using this project.
    ├── configs <- Setting files e.g. model parameter
    ├── data
    │   ├── features    <- Intermediate data.
    │   ├── input   <- Raw data.
    │   ├── logs    <- Log files
    │   └── output  <- Output data.
    ├── docs    <- Document folder e.g. Sphinx docs
    ├── models  <- Trained models
    ├── requirements.txt    <- Requirement file for the environment.
    ├── setup.py    <- makes project pip installable (pip install -e .) so src can be imported.
    └── src <- Source code
        ├── __init__.py
        ├── crawler <- clawler for collect datasets.
        ├── experiments <- folder for experiment.
        │   ├── code    <- experimental code like Jupyter Notebook.
        │   ├── description.md  <- 
        │   └── reports <- Generated analysis reports as HTML, PDF, LaTeX, etc.
        │       └── figs    <- Generated figures and other files to be used in reporting
        ├── tests   <- Test code
        └── utils   <- Utility code

---

Project based on the cookiecutter
* Inspired cookiecutter data science project template. #cookiecutterdatascience