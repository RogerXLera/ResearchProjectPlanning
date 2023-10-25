# ResearchProjectPlanning
## Research Project Planning (RPP).
------------
This repository contains the implementation of RPP problem algorithm. The data considers real research projects of the [IIIA-CSIC](https://www.iiia.csic.es/en-us/). The names of the projects and researchers are random letters for the sake of anonymity. 
The main developers of this implemmentation are Roger X. Lera and [Filippo Bistaffa](https://filippobistaffa.github.io/).

Dependencies
----------
 - [Python 3](https://www.python.org/downloads/)
 - [Pandas](https://pandas.pydata.org/)
 - [Csv library](https://docs.python.org/3/library/csv.html)
 - [Numpy](https://numpy.org/)
 - [CVXPY](https://www.cvxpy.org/)
 - [CPLEX](https://www.ibm.com/es-es/products/ilog-cplex-optimization-studio)

Dataset
----------
The data considers real research projects of the [IIIA-CSIC](https://www.iiia.csic.es/en-us/). The names of the projects and researchers are random letters for the sake of anonymity. 

Execution
----------
Our approach must be executed by means of the [`solve.py`](solve.py) Python script, i.e.,
```
usage: python3 solve.py [-a A] [-b B] [-g G] [-m M] [--solver SOLVER] [--instance] [--robust]

optional arguments:
  -a A            alpha parameter (default: 0.8)
  -b B            beta parameter (default: 0.1)
  -g G            gamma parameter (default: 0.1)
  -m M            budget variability (default: 1.0)
  --solver SOLVER solver to compute the solution (choices=['CPLEX','GUROBI'])
  --instance      generate random instances of projects and researchers
  --robust        apply SRA to obtain robust plans considering uncertainty
```

