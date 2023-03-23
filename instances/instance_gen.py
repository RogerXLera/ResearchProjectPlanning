"""
Author: Roger Xavier Lera Leri
Date: 14/03/2023
In this script we will generate test instances of the RPP.
"""
import numpy as np
import random as rd
import time
import os
from definitions import *

def project_duration(W):
    """
    This function evaluates the project duration from the duration of its workpackages.
    """
    sd = np.inf
    fd = 0

    for w in W.keys():
        sd_w = W[w][0]
        dur_w = W[w][1]
        if sd_w < sd:
            sd = sd_w

        if sd_w + dur_w - 1 > fd:
            fd = sd_w + dur_w - 1

    return fd - sd + 1

def project_structure(N_proj,wp_min,wp_max,dur_mean,dur_sd,ded_l,lambda_w,lambda_p,cost_min,cost_max,mu=1.0):
    """
    This function generates the project structure.
    """

    P = []
    month_p = 1
    year_p = 2000
    wp_id = 0

    for p in range(1,N_proj+1):
        
        project = Project(id=p,name=f"P{p}")
        # update starting date 
        if p != 1: # start date of project p
            delta_m = rd.expovariate(lambda_p)
            month_p += round(delta_m)
            while month_p > 12:
                month_p = month_p - 12
                year_p += 1
        sd_p = Month(month=month_p,year=year_p)

        N_wp = rd.randint(wp_min,wp_max) # number of wp of project p

        month_w = sd_p.month
        year_w = sd_p.year
        for w in range(1,N_wp+1): # generate wps
            wp_id += 1
            dur_w = round(rd.gauss(mu=dur_mean,sigma=dur_sd))
            ded_w = round(rd.expovariate(lambd=ded_l))

            if w != 1: # start date of wp w
                delta_m = rd.expovariate(lambda_w)
                month_w += round(delta_m)
                while month_w > 12:
                    month_w = month_w - 12
                    year_w += 1
                
            sd_w = Month(month=month_w,year=year_w)
            end_month = sd_w.month + dur_w
            end_year = sd_w.year
            while end_month > 12:
                end_month = end_month - 12
                end_year += 1
            ed_w = Month(month=end_month,year=end_year)

            wp_ = WorkPackage(id_=wp_id,project=project,start_month=st_w,end_month=ed_w,dedication=ded_w)
            wp.add(project.wp)

        ded_p = project.dedication()
        project.budget = mu*ded_p*rd.uniform(cost_min,cost_max)
        #dur_p = project_duration(W)
        #P.update({p:(sd_p,dur_p,ded_p,bud_p,W)})
        P.append(project)
    
    return P


def researcher_structure(P,RS,cost_min,cost_max,av_min,av_max,rep):
    """
    This function generates the researchers structure.
    """
    R = {}
    r = 1

    for p in P.keys():
        
        d_p = P[p][2] # dedication of project p
        d_p_prima = 0
        sd_p = P[p][0]
        dur_p = P[p][1]
        while d_p > d_p_prima:
            wl = 0
            create_res = False
            n_res = len(R.keys())
            if n_res > rep:
                rep_ = rep
            else:
                rep_ = n_res

            for i in range(0,rep_):
                res = rd.randint(1,n_res)
                wl_r = 0
                proj_inv = list(R[res][2].keys())
                if p not in proj_inv:
                    
                    for proj in R[res][2].keys():
                        t_p = 0 # eval target per project p of researcher r
        
                        for m in range(sd_p,sd_p+dur_p):
                            try:
                                target_m = R[res][2][proj][m]
                            except:
                                target_m = 0
                            t_p += target_m

                        
                        wl_r += (t_p/R[res][1])/dur_p

                    #if 1 - np.exp(-wl_r/RS) < rd.random():
                    if rd.random() < RS - wl_r:
                        #print(f"Involve Res r: {res}")
                        t_p = 0
                        target = {}
                        t_p_ = rd.uniform(0,1-wl_r)
                        for m in range(sd_p,sd_p+dur_p):
                            #rand = rd.randint(0,R[res][1]-round(wl_r*R[res][1]))
                            #t_p += rand
                            #target.update({m:rand})
                            t_p += round(R[res][1]*t_p_)
                            target.update({m:round(R[res][1]*t_p_)})

                        R[res][2].update({p:target})
                        create_res = True
                        break

            if create_res == False:
                #print("New res: ", r)
                cost_r = rd.uniform(cost_min,cost_max)
                av_r = rd.randint(av_min,av_max)
                target = {}
                t_p = 0
                t_p_ = rd.random()
                for m in range(sd_p,sd_p+dur_p):
                    #rand = rd.randint(0,av_r)
                    #t_p += rand
                    #target.update({m:rand})
                    t_p += round(av_r*t_p_)
                    target.update({m:round(av_r*t_p_)})

                R.update({r:[cost_r,av_r,{p:target}]})
                r += 1

            d_p_prima += t_p

    return R


def RPP_instance(N_proj,wp_min,wp_max,dur_mean,dur_sd,ded_l,lambda_w,lambda_p,cost_min,cost_max,RS,costr_min,costr_max,av_min,av_max,rep=10,mu=1.0):
            
    P = project_structure(N_proj,wp_min,wp_max,dur_mean,dur_sd,ded_l,lambda_w,lambda_p,cost_min,cost_max,mu=1.0)
    R = researcher_structure(P,RS,costr_min,costr_max,av_min,av_max,rep)
    
    return P,R





if __name__ == '__main__':

    #rd.seed(0)
    st = time.time()
    P,R = RPP_instance(2,1,9,38.07,4.70,0.000757,1000,0.179,26.35,44.33,0.39,17.16,54.01,130,130,rep=3)
    ft = time.time()
    print(f"Instance generation time: {ft-st}")
    print("N res: ", len(R.keys()))

    for r in R.keys():
        print(f"Researcher ID: {r}, # of projects: {len(R[r][2].keys())}")