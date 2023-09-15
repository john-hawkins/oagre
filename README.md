# OaGRe

```
Status: Functional - 
```

A meta-learning model for regression on noisy data with outliers

This work was initially started in 2017 while working with a large scale
noisy regression problem. The initial experiments were done in R and abandoned
when I moved onto other projects. The same line of thinking recurred in 2021
as I looked at more regression problems and led to this repository.

This time round the implementation has been done in Python in a scikit-learn 
compatible structure. It also allows you to define the internal classifier and 
regression algorithms to be used, rather than forcing the use of decision trees.

Massive thanks are required to the contributors at the scikit-lego project for 
the inspiring open-source library that informed much of the development here.


### Installation

The package will be released via PyPi and can installed via pip.

Alternatively you can install from source code

### Experiments

We have conducted experiments using synthetically generated data with highly non-linear 
regression problems and heteroscedastic noise. Executed using the script [scripts/run_experiment.py]
and the analysed with the script [scripts/analyse.py].

## How to cite

Paper Submitted to AICI 2024 - Awaiting Review

```bibtex
@InProceedings{Hawkins2024,
   author = {John Hawkins},
   year = {2024},
   title = {OAGRE: Outlier Attenuated Gradient Boosted Regression},
   booktitle = {},
   month = {},
}
```

