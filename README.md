# COP

COP is a Python package for solving the convex optimization problem (COP) in the form:

```
inf f0(x)
s.t. fi(x) <= 0, i=1,...,m
     Ax=b
```
# Getting started

1. Clone this repository: git clone https://github.com/dinhthilan/COP.git 
2. Now, open a command line. Assuming COP is in your current folder,

Activate your python enviroment

```
conda activate python3.9
```
Install the dependencies: 
```
pip install requirements.txt
```
3. To run an experiment, 
```
cd COP/TESTPROB/

```
Then type
```

python test_Rand_Prob1_10.py     # Case 1: Table 4: Id 1 and Figure 1
python test_Rand_Prob1_100.py    # Case 1: Table 4: Id 2 and Figure 2 
python test_Rand_Prob1_1000.py   # Case 1: Table 4: Id 3 and Figure 3 
python test_Rand_Prob2_10.py        # Case 2: Table 4: Id 4 and Figure 4
python test_Rand_Prob2_100.py        # Case 2: Table 4: Id 5 and Figure 5
python test_Rand_Prob2_1000.py        # Case 2: Table 4: Id 6 and Figure 6 

python test_MAD8.py	          # Table 5, 6: Id 7 and Figure 7 
python test_Wong2.py             # Table 5, 6: Id 8 and Figure 8 
python test_Wong3.py             # Table 5, 6: Id 9 and Figure 9 

python test_LAD10.py        	  # Table 7, 8: Id 10 and Figure 10
python test_LAD100.py       # Table 7, 8: Id 11 and Figure 11
python test_LAD1000.py      # Table 7, 8: Id 12 and Figure 12

python test_SVM2.py	     # Table 9, 10: Id 13 and Figure 13
python test_SVM3.py	     # Table 9, 10: Id 14 and Figure 14
python test_SVM5.py	     # Table 9, 10: Id 15 and Figure 15

```

# References
For more details, please refer to:
T.L. Dinh, N.H.A. Mai. [Comparing different subgradient methods  for solving convex optimization problems with functional constraints](https://arxiv.org/abs/2101.01045), 2021. Submitted.

