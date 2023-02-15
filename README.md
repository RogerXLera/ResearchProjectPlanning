# ResearchProjectPlanning
## Research Project Planning (RPP).
===================
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
usage: solve.py [-h] [-m M] [-t T] [--alpha ALPHA] [-p P] [-j J] [-a A] [--name NAME] [--model MODEL] 
                [-- solver SOLVER]

optional arguments:
  -h, --help      show this help message and exit
  -m M            number of time periods (default: 100)
  -t T            tau dedication per period (default: 2)
  --alpha ALPHA   cost function weight parameter (default: 0.5)
  -p P            p-norm (default: 1)
  -j J            Job ID 
  -a A            activities file
  --name NAME     user name
  --model MODEL   ILP encoding model (choices=['fixed','variable','mixed'])
  --solver SOLVER solver to compute the solution (choices=['CPLEX','GUROBI'])
```

