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

def project_structure(N_proj,wp_min,wp_max,dur_mean,dur_sd,ded_l,lambda_w,lambda_p,cost_min,cost_max):
    """
    This function generates the project structure.
    """

    P = []
    month_p = 1
    year_p = 2000

    for p in range(1,N_proj+1):
        
        project = Project(id_=p,name=f"P{p}")
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
        wp_id = 0
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

            wp_ = WorkPackage(id_=wp_id,project=project,start_month=sd_w,end_month=ed_w,dedication=ded_w)
            wp_.add(project.wp)

        sd_p,ed_p = project.date() # starting date and endind date
        per_p = Period(id_=0,start=sd_p,end=ed_p)
        per_p.add(project.period)
        ded_p = project.dedication()
        project.budget = ded_p*rd.uniform(cost_min,cost_max)
        project.add(P)
    
    return P


def researcher_structure(P,RS,cost_min,cost_max,av_min,av_max,rep):
    """
    This function generates the researchers structure.
    """
    R = []
    r = 1

    for p in P:
        
        d_p = p.dedication() # dedication of project p
        d_p_prima = 0 # current project dedication satisfied
        sd_p,ed_p = p.date() # starting date and endind date
        per_p = Period(id_=0,start=sd_p,end=ed_p)
        dur_p = per_p.duration()
        while d_p > d_p_prima:
            wl = 0
            create_res = False
            n_res = len(R)
            if n_res > rep:
                rep_ = rep
            else:
                rep_ = n_res

            for i in range(0,rep_):
                r_ = rd.randint(0,n_res-1)
                res = R[r_] #Researcher object
                wl_r = 0
                if p not in res.projects:
                    
                    for proj in res.projects:
                        t_p = 0 # eval target per project p of researcher r

                        cm = sd_p # current month
                        m = 0
                        while m < dur_p:
                            m += 1
                            for target in proj.target:
                                if target.researcher == res and target.period.check(cm):
                                    t_p += target.value
                                    break
                            if cm.month == 12:
                                cm = Month(month=1,year=cm.year+1)
                            else:
                                cm = Month(month=cm.month+1,year=cm.year)
                        
                        wl_r += t_p/dur_p

                    if rd.random() < RS - wl_r:
                        t_p_ = rd.uniform(0,1-wl_r)
                        for period in p.period:
                            t_p = Target(project=p,researcher=res,period=period,value=t_p_)
                            t_p.add(p.target)

                        res.add(p.researchers)
                        res.projects.append(p)
                        create_res = True
                        break

            if create_res == False:
                cost_r = rd.uniform(cost_min,cost_max)
                av_r = rd.randint(av_min,av_max)
                con_r = rd.choice([True, False])
                res = Researcher(id_=r,name=f"R{r}",cost=cost_r,time=av_r,contract=con_r)
                t_p_ = rd.random()
                for period in p.period:
                    t_p = Target(project=p,researcher=res,period=period,value=t_p_)
                    t_p.add(p.target)

                res.add(R)
                res.add(p.researchers)
                res.projects.append(p)
                r += 1

            d_p_prima += t_p_*res.time*dur_p

    return R


def RPP_instance(N_proj,wp_min,wp_max,dur_mean,dur_sd,ded_l,lambda_w,lambda_p,cost_min,cost_max,RS,costr_min,costr_max,av_min,av_max,rep=10):
            
    P = project_structure(N_proj,wp_min,wp_max,dur_mean,dur_sd,ded_l,lambda_w,lambda_p,cost_min,cost_max)
    R = researcher_structure(P,RS,costr_min,costr_max,av_min,av_max,rep)
    
    return P,R





if __name__ == '__main__':

    rd.seed(0)
    st = time.time()
    P,R = RPP_instance(2,1,9,38.07,4.70,0.000757,1000,0.179,26.35,44.33,0.39,17.16,54.01,130,130,rep=3)
    ft = time.time()
    print(f"Instance generation time: {ft-st}")
    print("N res: ", len(R))
    for r in R:
        print(r)

    for p in P:
        print(p)
        for w in p.wp:
            print(f"Name: {w.name} \t Dedication: {w.dedication} \t Start: {w.start} \t End: {w.end}")
        for t in p.target:
            print(t)
        for r in p.researchers:
            print(r.name)
        for pe in p.period:
            print(pe)
        sd,ed = p.date()
        print(f"Starting month of the project, {p.name}: {sd}")
        print(f"Ending month of the project, {p.name}: {ed}")
    
    