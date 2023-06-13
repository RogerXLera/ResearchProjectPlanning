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
from generate_instance import RPP_instance
from process_instance import matrices
import random as rd
from generate_uncertainty import RPP_uncertainty
from process_uncertainty import matrices_u


def definitions_generation(res_file_path,projects_folder):
    """
    This function generates the definitions of the formalisation
    """
    R = read_researcher(res_file_path)
    P = instance_projects(projects_folder,R)

    return P,R


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


def ilp_model_u(variables,matrices,model_par):
    """
    This function returns the cp.problem object and the model variables
    """
    xp,xw = variables
    A_list,t_list,D_list,d_list,T_list,tau_list,B_list,b_list = matrices
    alpha,beta,gamma,mu,w_rho,w_delta,prob_u = model_par
    n_uncertainty = len(A_list) - 1
    # let's weight the probability of each instance to happen
    probability = np.ones(n_uncertainty+1)
    probability[0] *= prob_u
    probability[1:] *= (1-prob_u)/n_uncertainty
    

    len_x = len(xw)
    ############################VARIABLES#############################
    
    x = cp.Variable(len_x,integer=False) # set variables as a vector x
    u_list = []
    v_list = []
    #u = cp.Variable(len_b*n_uncertainty,integer=False) # set variables as a vector u
    #v = cp.Variable(len_b*n_uncertainty,integer=False) # set variables as a vector v
    A = cp.Variable(n_uncertainty+1,integer=False) # set variables as a vector A
    B = cp.Variable(n_uncertainty+1,integer=False) # set variables as a vector B
    G = cp.Variable(n_uncertainty+1,integer=False) # set variables as a vector \Gamma
    #rho = cp.Variable(n_uncertainty,integer=False) # set variables as a vector \rho
    #delta = cp.Variable(n_uncertainty,integer=False) # set variables as a vector \delta
    rho_list = []
    delta_list = []
    RHO = cp.Variable(n_uncertainty,integer=False)
    DELTA = cp.Variable(n_uncertainty,integer=False)

    ############################OBJECTIVE#############################

    cost = alpha*probability@A + beta*probability@B + gamma*probability@G + w_rho*probability[1:]@RHO + w_delta*probability[1:]@DELTA
    ###########################CONSTRAINTS###########################
    cs = [x >= np.zeros_like(x,dtype=int)]
    for i in range(n_uncertainty+1):
        
        len_b = np.shape(B_list[i])[0]
        u = cp.Variable(len_b,integer=False)
        u_list += [u]
        v = cp.Variable(len_b,integer=False)
        v_list += [v]
        cs += [cp.sum_squares(A_list[i]@x - t_list[i]) <= A[i]]
        cs += [cp.sum_squares(v) <= B[i]]
        cs += [cp.sum_squares(u) <= G[i]]
        cs += [B_list[i] @ x - mu*b_list[i] <= u]
        cs += [B_list[i] @ x - mu*b_list[i] >= v]
        cs += [np.zeros_like(b_list[i],dtype=int) <= u]
        cs += [np.zeros_like(b_list[i],dtype=int) >= v]
        if i == 0:   
            cs += [D_list[i] @ x >= d_list[i]] #dedication constraint
            cs += [T_list[i] @ x <= tau_list[i]] #staff dedication constraint
        else:
            len_rho = len(tau_list[i])
            rho = cp.Variable(len_rho,integer=False)
            rho_list += [rho]

            len_del = len(d_list[i])
            delta = cp.Variable(len_del,integer=False)
            delta_list += [delta]

            cs += [D_list[i] @ x >= d_list[i] - delta] #dedication constraint
            cs += [T_list[i] @ x <= tau_list[i] + rho] #staff dedication constraint
            cs += [delta >= np.zeros_like(delta,dtype=int)]
            cs += [rho >= np.zeros_like(rho,dtype=int)]
            cs += [cp.sum_squares(delta) <= DELTA[i-1]]
            cs += [cp.sum_squares(rho) <= RHO[i-1]]

    ######################PROBLEM###############################
    
    
    prob = cp.Problem(cp.Minimize(cost),cs)
    return prob,x,u_list,v_list,A,B,G,rho_list,delta_list,RHO,DELTA


def solve_problem(problem,x,u,v):
    """
        We solve the problem
    """
    start_time = time.time()
    problem.solve(solver=args.solver,verbose=True,cplex_params={"barrier.limits.objrange":1e20,"barrier.display":1,
                                                                #"barrier.limits.iteration":1e3,
                                                                })
    finish_time = time.time()
    obj_value = problem.value
    print("The optimal value is", obj_value)
    xx = np.array(x.value) #list with the results of x (|x| dim)
    uu = np.array(u.value)
    vv = np.array(v.value)

    print(f"Time: {finish_time - start_time:.2f}")
    return obj_value,xx,uu,vv

def solve_problem_u(problem,x,u,v,A,B,G,rho,delta,RHO,DELTA):
    """
        We solve the problem
    """
    n_uncertainty = len(rho)
    start_time = time.time()
    problem.solve(solver=args.solver,verbose=True,cplex_params={"barrier.limits.objrange":1e20,"barrier.display":1,
                                                                #"barrier.limits.iteration":1e3,
                                                                })
    finish_time = time.time()
    obj_value = problem.value
    print("The optimal value is", obj_value)
    xx = np.array(x.value) #list with the results of x (|x| dim)
    AA = np.array(A.value) 
    BB = np.array(B.value) 
    GG = np.array(G.value) 
    RRHO = np.array(RHO.value) 
    DDELTA = np.array(DELTA.value) 
    uu = [np.array(u[i].value) for i in range(n_uncertainty+1)]
    vv = [np.array(v[i].value) for i in range(n_uncertainty+1)]
    rrho = [np.array(rho[i].value) for i in range(n_uncertainty)]
    ddelta = [np.array(delta[i].value) for i in range(n_uncertainty)]

    print(f"Time: {finish_time - start_time:.2f}")

    return obj_value,xx,uu,vv,AA,BB,GG,rrho,ddelta,RRHO,DDELTA

def residual(vec,norm_vec):

    vec_sum = np.sum(np.absolute(vec))
    norm_vec_sum = np.sum(np.absolute(norm_vec))

    return vec_sum/norm_vec_sum*100
    
def contract_dedication(xw,xx):

    con_ded = 0
    for i in range(len(xw)):
        r = xw[i][0]
        if r.contract:
            con_ded += xx[i]
    
    total_ded = np.sum(np.absolute(xx))

    return con_ded/total_ded*100


def print_metrics(value,variables,matrices,params):

    xp,xw,xx,uu,vv = variables
    A,t,D,d,T,tau,B,b = matrices
    alpha,beta,gamma,mu = params
    fairness = residual(A@xx-t,t)
    ded_residual = residual(D@xx-d,d)
    bud_residual = residual(B@xx -b,b)
    res_residual = residual(T@xx - tau,tau)
    con_ded = contract_dedication(xw,xx)
    bud_eff = np.sum(np.absolute(vv))
    bud_exc = np.sum(np.absolute(uu))
    target = np.sum(np.absolute(t))

    print(f"######################################")
    print(f"############# Results ################")
    print(f"######################################")
    print(f"Alpha, Beta, Gamma, Mu, Fairness, Dedication R, Budget R, Researchers R, Contract d, Budget eff, Budget exc, Target")
    print(f"{alpha:.7f},{beta:.7f},{gamma:.7f},{mu:.3f},{fairness:.3f},{ded_residual:.3f},{bud_residual:.3f},{res_residual:.3f},{con_ded:.3f},{bud_eff:.3f},{bud_exc:.3f},{target:.3f}")
    return None


def print_metrics_u(value,variables,matrices,params):

    xp,xw,xx,uu,vv,AA,BB,GG,rrho,ddelta,RRHO,DDELTA = variables
    A_list,t_list,D_list,d_list,T_list,tau_list,B_list,b_list = matrices
    n_uncertainty = len(rrho)
    alpha,beta,gamma,mu,w_rho,w_delta,w_prob = params
    matrices_ = A_list[0],t_list[0],D_list[0],d_list[0],T_list[0],tau_list[0],B_list[0],b_list[0]
    variables_ = xp,xw,xx,uu[0],vv[0]
    print_metrics(value,variables_,matrices_,params[:4])
    AA_ = np.sum(np.absolute(AA))
    BB_ = np.sum(np.absolute(BB))
    GG_ = np.sum(np.absolute(GG))
    RRHO_ = np.sum(np.absolute(RRHO))
    DDELTA_ = np.sum(np.absolute(DDELTA))

    print(f"N instances, W rho, W delta, W prob, A, B, G, RHO, DELTA")
    print(f"{n_uncertainty},{w_rho:.3f},{w_delta:.3f},{w_prob:.3f},{AA_:.3f},{BB_:.3f},{GG_:.3f},{RRHO_:.3f},{DDELTA_:.3f}")

    return None

def print_instance_metrics(P,R,M):
    
    len_p = len(P)
    len_r = len(R)
    len_m = len(M.sequence)

    print(f'################################################')
    print(f'################### METRICS ####################')
    print(f'################################################')
    print("Total Projects, Total Researchers, Total Months")
    print(f"{len_p},{len_r},{len_m}")
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
    parser.add_argument('-v', type=int, default=130, help='v: Minimum hours available per month and researcher')
    parser.add_argument('-V', type=int, default=130, help='V: Maximum hours available per month and researcher')
    parser.add_argument('-e', type=int, default=3, help='e: Number of repetitions to search an alternative researcher in researchers instance generator.')
    parser.add_argument('-m', type=float, default=1.0, help='m: Budget scarcity')
    parser.add_argument('-a', type=float, default=0.8, help='a: Alpha parameter')
    parser.add_argument('-b', type=float, default=0.1, help='b: Beta parameter')
    parser.add_argument('-g', type=float, default=0.1, help='g: Gamma parameter')
    parser.add_argument('--rho', type=float, default=0.5, help='b: Beta parameter')
    parser.add_argument('--delta', type=float, default=0.5, help='g: Gamma parameter')
    parser.add_argument('-p', type=float, default=0.5, help='p: Probability of not receiving uncertainty')
    parser.add_argument('--researchers', type=str, default='researchers.csv', help='--researchers: Researchers file')
    parser.add_argument('--projects', type=str, default='projects', help='--projects: projects folder')
    parser.add_argument('--seed', type=int, default=0, help='--seed: Seed')
    parser.add_argument('--solver', type=str, choices=['CPLEX', 'GUROBI'], default='CPLEX')
    parser.add_argument('--file', help='--file: Store the results in a txt file. -f for filename', action='store_true')
    parser.add_argument('--instance', help='--instance: Create an instance', action='store_true')
    parser.add_argument('-k', type=int, default=10, help='k: Number of instances')
    parser.add_argument('--nproj', type=int, default=3, help='nproj: Number of projects added')
    parser.add_argument('--robust', help='--robust: Apply a robust approximation', action='store_true')
    parser.add_argument('-I', type=int, default=10, help='I: Number of instances')
    parser.add_argument('-f', type=str, default="results", help='f: Name of the file')
    args = parser.parse_args()
    seed_ = args.seed
    rd.seed(seed_)

    researchers_file_path = os.path.join(path_,wd_,args.researchers)
    projects_folder = os.path.join(path_,wd_,args.projects)
    param_ = (args.w,args.W,args.l,args.L,args.d,args.y,args.Y,args.c,args.C,args.S,args.r,args.R,args.v,args.V,args.e)
    
    ## data
    if args.instance:
        P,R = RPP_instance(args.N,args.w,args.W,args.l,args.L,args.d,args.y,args.Y,args.c,args.C,args.S,args.r,args.R,args.v,args.V,args.e)
    else:
        P,R = definitions_generation(researchers_file_path,projects_folder)
    
    if args.robust:
        I = RPP_uncertainty(P,R,args.k,args.nproj,param_)


    if args.robust:
        data = matrices_u(P,R,I)
        M_total,xp_total,xw_total,A_list,t_list,D_list,d_list,T_list,tau_list,B_list,b_list = data
        print_instance_metrics(P,R,M_total)
        #model
        model_par = (args.a,args.b,args.g,args.m,args.rho,args.delta,args.p)
        variables = (xp_total,xw_total)
        matrices_ = (A_list,t_list,D_list,d_list,T_list,tau_list,B_list,b_list)
        prob,x,u,v,A,B,G,rho,delta,RHO,DELTA = ilp_model_u(variables,matrices_,model_par)

        #solve
        value,xx,uu,vv,AA,BB,GG,rrho,ddelta,RRHO,DDELTA = solve_problem_u(prob,x,u,v,A,B,G,rho,delta,RHO,DELTA)
        variables_ = xp_total,xw_total,xx,uu,vv,AA,BB,GG,rrho,ddelta,RRHO,DDELTA
        print_metrics_u(value,variables_,matrices_,model_par)
    
    else:
        data = matrices(P,R)
        M,xp,xw,A,t,D,d,T,tau,B,b = data
    
        #model
        model_par = (args.a,args.b,args.g,args.m)
        variables = (xp,xw)
        matrices_ = (A,t,D,d,T,tau,B,b)
        prob,x,u,v = ilp_model(variables,matrices_,model_par)

        #solve
        value,xx,uu,vv = solve_problem(prob,x,u,v)
        variables_ = xp,xw,xx,uu,vv
        print_metrics(value,variables_,matrices_,model_par)


    