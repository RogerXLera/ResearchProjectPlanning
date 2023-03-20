"""
Roger Lera
2023/03/09
"""
import os
import cvxpy as cp
import numpy as np
import argparse as ap
import time
import csv
from definitions import * # classes Skill, Activity, TimePeriod, TimePeriodSequence, Job, User Preference
from read_instance import read_researcher,instance_projects
from process_instance import *
import random as rd


def definitions_generation(res_file_path,projects_folder):
    """
    This function generates the definitions of the formalisation
    """
    R = read_researcher(res_file_path)
    P = instance_projects(projects_folder,R)
    data = matrices(P,R)
    M,xp,xw,A,t,D,d,T,tau,B,b = data

    return R,P,M,xp,xw,A,t,D,d,T,tau,B,b


def ilp_model(variables,matrices,model_par):
    """
    This function returns the cp.problem object and the model variables
    """
    xp,xw = variables
    A,t,D,d,T,tau,B,b = matrices
    alpha,beta,gamma,mu = model_par

    len_x = len(xw)
    len_b = np.shape(B)[0]
    ############################VARIABLES#############################
    
    x = cp.Variable(len_x,integer=False) # set variables as a vector x
    u = cp.Variable(len_b,integer=False) # set variables as a vector x
    v = cp.Variable(len_b,integer=False) # set variables as a vector x

    ############################OBJECTIVE#############################

    term1 = cp.sum_squares(A @ x - t) #set the first term 
    term2 = cp.sum_squares(v)
    term3 = cp.sum_squares(u)

    ###########################CONSTRAINTS###########################
    c1 = [x >= np.zeros_like(x,dtype=int)]
    c2 = [D @ x >= d] #dedication constraint
    c3 = [T @ x <= tau] #staff dedication constraint
    c4 = [B @ x - mu*b <= u]
    c5 = [B @ x - mu*b >= v] 
    c6 = [np.zeros_like(b,dtype=int) <= u]
    c7 = [np.zeros_like(b,dtype=int) >= v]
    #########COST FUNCTION########################################
    
    cs = c1+c2+c3+c4+c5+c6+c7
    
    prob = cp.Problem(cp.Minimize(alpha*term1+beta*term2+gamma*term3),cs)
    return prob,x,u,v


def solve_problem(problem,x,u,v):
    """
        We solve the problem
    """
    start_time = time.time()
    problem.solve(solver=args.solver,verbose=True,cplex_params={})
    finish_time = time.time()
    obj_value = problem.value
    print("The optimal value is", obj_value)
    xx = np.array(x.value) #list with the results of x (|x| dim)
    uu = np.array(u.value)
    vv = np.array(v.value)

    print(f"Time: {finish_time - start_time:.2f}")
    
    return obj_value,xx,uu,vv

def print_analytics(z,x,B,A,P,t):
    """
    Print the analytics of the Results
    """
    ja = job_affinity(t,z,args.p)
    print(f"Job Affinity: {ja} (%)")
    money_period = B@x
    len_p = len(P.sequence)
    for i in range(len_p):
        print(f"Period: {P.sequence[i].id} \t Budget: {money_period[i]} (ZLTOs)")

    return None
        

if __name__ == '__main__':

    wd_ = 'data'
    path_ = os.getcwd()
    parser = ap.ArgumentParser()
    parser.add_argument('-N', type=int, default=10, help='N: Number of projects')
    parser.add_argument('-w', type=int, default=1, help='w: Minimum # of work packages per project')
    parser.add_argument('-W', type=int, default=9, help='W: Maximum # of work packages per project')
    parser.add_argument('-l', type=float, default=38.07, help='l: Mean duration of a work package (months)')
    parser.add_argument('-L', type=float, default=4.70, help='L: Standard deviation of the duration of a work package (months)')
    parser.add_argument('-d', type=float, default=0.000757, help='d: Exponential distribution parameter of the dedication in a work package (1/hours)')
    parser.add_argument('-y', type=float, default=1000.0, help='y: Lambda parameter of the exponential distribution of work packages time difference (1/months). Represents the frequency of work packages starting within the same project.')
    parser.add_argument('-Y', type=float, default=0.179, help='Y: Lamda parameter of the exponential distribution of projects time difference (1/months). Represents the frequency of projects starting.')
    parser.add_argument('-c', type=float, default=26.35, help='c: Minimum mean cost of a project (euros/hour)')
    parser.add_argument('-C', type=float, default=44.33, help='C: Maximum mean cost of a project (euros/hour)')
    parser.add_argument('-S', type=float, default=0.39, help='S: Resource Strength in (0,1]: RS is interpreted as the capacity of researchers to dedicate all their time in the projects. As higher RS, more work load they will have.')
    parser.add_argument('-r', type=float, default=17.16, help='r: Minimum mean cost of a researcher (euros/hour)')
    parser.add_argument('-R', type=float, default=54.01, help='R: Maximum mean cost of a researcher (euros/hour)')
    parser.add_argument('-v', type=int, default=120, help='v: Minimum hours available per month and researcher')
    parser.add_argument('-V', type=int, default=140, help='V: Maximum hours available per month and researcher')
    parser.add_argument('-e', type=int, default=3, help='e: Number of repetitions to search an alternative researcher in researchers instance generator.')
    parser.add_argument('-m', type=float, default=1.0, help='m: Budget scarcity')
    parser.add_argument('-a', type=float, default=1.0, help='a: Alpha parameter')
    parser.add_argument('-b', type=float, default=1.0, help='b: Beta parameter')
    parser.add_argument('-g', type=float, default=1.0, help='g: Gamma parameter')
    parser.add_argument('--researchers', type=str, default='researchers.csv', help='--researchers: Researchers file')
    parser.add_argument('--projects', type=str, default='projects', help='--projects: projects folder')
    parser.add_argument('--seed', type=int, default=0, help='--seed: Seed')
    parser.add_argument('--solver', type=str, choices=['CPLEX', 'GUROBI'], default='CPLEX')
    parser.add_argument('--file', help='--file: Store the results in a txt file. -f for filename', action='store_true')
    parser.add_argument('-f', type=str, default="results", help='f: Name of the file')
    args = parser.parse_args()
    seed_ = args.seed
    rd.seed(seed_)

    researchers_file_path = os.path.join(path_,wd_,args.researchers)
    projects_folder = os.path.join(path_,wd_,args.projects)
    
    #data
    R,P,M,xp,xw,A,t,D,d,T,tau,B,b = definitions_generation(researchers_file_path,projects_folder)
    
    #model
    model_par = (args.a,args.b,args.g,args.m)
    variables = (xp,xw)
    matrices = (A,t,D,d,T,tau,B,b)
    prob,x,u,v = ilp_model(variables,matrices,model_par)

    #solve
    value,xx,uu,vv = solve_problem(prob,x,u,v)


    