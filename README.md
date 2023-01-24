# COP

COP is a Python package for solving the convex optimization problem (COP) in the form:

```
inf f0(x)
s.t. fi(x) <= 0, i=1,...,m
     Ax=b
```
# Getting started

1. Clone this repository: https://github.com/dinhthilan/COP.git
2. Install the dependencies: pip install requirements.txt
3. Now, to run an experiment, open a command line. Assuming COP is in your current folder,
```
cd COP/TESTPROB/

```
Activate your python enviroment

```
conda activate python3.6
```
Then type
```
python experiment_file_name
```

# References
For more details, please refer to:
T.L. Dinh, N.H.A. Mai. [Comparing different subgradient methods  for solving convex optimization problems with functional constraints](https://arxiv.org/abs/2101.01045), 2021. Submitted.

