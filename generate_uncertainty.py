"""
Author: Roger Xavier Lera Leri
Date: 8/06/2023
In this script we will generate artificial instances of the RPP.
"""
import numpy as np
import random as rd
import time
import os
from definitions import *

def project_structure(P,N_proj,wp_min,wp_max,dur_mean,dur_sd,ded_l,lambda_w,lambda_p,cost_min,cost_max):
    """
    This function generates the project structure.
    """
    len_p = len(P)
    month_p = P[0].date()[0].month
    year_p = P[0].date()[0].year
    P_new = []

    for p in range(len_p+1,len_p+N_proj+1):
        
        project = Project(id_=p,name=f"P{p}")

        # update starting date
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
        project.add(P_new)
    
    return P_new


def researcher_structure(P,R,len_r,RS,cost_min,cost_max,av_min,av_max,rep):
    """
    This function generates the researchers structure.
    """

    r = len_r+1
    R_i = R.copy()
    R_new = []

    for p in P:
        
        d_p = p.dedication() # dedication of project p
        d_p_prima = 0 # current project dedication satisfied
        sd_p,ed_p = p.date() # starting date and endind date
        per_p = Period(id_=0,start=sd_p,end=ed_p)
        dur_p = per_p.duration()
        while d_p > d_p_prima:
            wl = 0
            create_res = False
            n_res = len(R_i)
            if n_res > rep:
                rep_ = rep
            else:
                rep_ = n_res

            for i in range(0,rep_):
                r_ = rd.randint(0,n_res-1)
                res = R_i[r_] #Researcher object
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
                if cost_r < cost_min + (cost_max-cost_min)/2:
                    con_r = True
                else:
                    con_r = False
                res = Researcher(id_=r,name=f"R{r}",cost=cost_r,time=av_r,contract=con_r)
                t_p_ = rd.random()
                for period in p.period:
                    t_p = Target(project=p,researcher=res,period=period,value=t_p_)
                    t_p.add(p.target)

                res.add(R_i)
                res.add(R_new)
                res.add(p.researchers)
                res.projects.append(p)
                r += 1

            d_p_prima += t_p_*res.time*dur_p

    return R_new


def researcher_structure_no_researchers(P,R):
    """
    This function generates the researchers structure.
    """
    n_res = len(R)
    for p in P:
        
        d_p = p.dedication() # dedication of project p
        d_p_prima = 0 # current project dedication satisfied
        sd_p,ed_p = p.date() # starting date and endind date
        per_p = Period(id_=0,start=sd_p,end=ed_p)
        dur_p = per_p.duration()
        while d_p > d_p_prima:
            wl = 0
            
            r_ = rd.randint(0,n_res-1)
            res = R[r_] #Researcher object
            t_p_ = 0
            if res not in p.researchers:
                t_p_ = rd.uniform(0,1)
                for period in p.period:
                    t_p = Target(project=p,researcher=res,period=period,value=t_p_)
                    t_p.add(p.target)

                res.add(p.researchers)
                res.projects.append(p)

            d_p_prima += t_p_*res.time*dur_p

    return None

def instance_objects(P,R,instance_id):

    if instance_id == 0:
        for p in P:
            p.instances.append(instance_id)
        for r in R:
            r.instances.append(instance_id)
        
        return None

    else:
        for p in P:
            if len(p.instances) > 0:
                if p.instances[0] == 0:
                    p.instances.append(instance_id)
            else:
                p.instances.append(instance_id)
        for r in R:
            if len(r.instances) > 0:
                if r.instances[0] == 0:
                    r.instances.append(instance_id)
            else:
                r.instances.append(instance_id)
        return None

def RPP_uncertainty(P,R,n_instances,N_max,param_):
            
    wp_min,wp_max,dur_mean,dur_sd,ded_l,lambda_w,lambda_p,cost_min,cost_max,RS,costr_min,costr_max,av_min,av_max,rep = param_
    
    instance_objects(P,R,0)
    instance_0 = Instance(0,P.copy(),R.copy())
    I = [instance_0]
    for i in range(1,n_instances+1):
        
        n_proj = rd.randint(1,N_max)
        P_new = project_structure(P,n_proj,wp_min,wp_max,dur_mean,dur_sd,ded_l,lambda_w,lambda_p,cost_min,cost_max)
        R_new = instance_0.researchers.copy()
        # adding new researchers
        #R_new = researcher_structure(P_new,instance_0.researchers,len(R),RS,costr_min,costr_max,av_min,av_max,rep)
        #without adding new researchers
        researcher_structure_no_researchers(P_new,R_new)
        P += P_new
        # adding new researchers
        #R += R_new       
        instance_objects(P,R,i)
        #adding new researchers
        #instance_i = Instance(i,instance_0.projects + P_new,instance_0.researchers + R_new)
        #without adding new researchers
        instance_i = Instance(i,instance_0.projects + P_new,R_new)
        I.append(instance_i)
    
    return I


def projects_print(P):
    for p in P:
        string = f"Project: {p.name} \t Instances: "
        for i in p.instances:
            string += f"{i} "
        print(string)
    return None


def researchers_print(R):
    for r in R:
        string = f"Researcher: {r.name} \t Instances: "
        for i in r.instances:
            string += f"{i} "
        print(string)
    return None


if __name__ == '__main__':

    rd.seed(0)
    st = time.time()
    print("---------------MAIN------------------")
    from read_instance import read_researcher,instance_projects
    path_ = os.getcwd()
    directory_ = 'data'
    file_ = 'researchers.csv'
    file_path = os.path.join(path_,directory_,file_)
    file_projects = os.path.join(path_,directory_,'projects')
    R = read_researcher(file_path)
    P = instance_projects(file_projects,R)
    ft = time.time()
    print(f"Instance generation time: {ft-st}")
    #projects_print(P)
    #researchers_print(R)
    param_ = (1,9,38.07,4.70,0.000757,1000,0.179,26.35,44.33,0.39,17.16,54.01,130,130,3)
    n_instances = 2
    N_max = 3
    I = RPP_uncertainty(P,R,n_instances,N_max,param_)
    ft2 = time.time()
    print(f"Instance generation time: {ft2-ft}")
    #projects_print(P)
    #researchers_print(R)

    for i in I:
        print(i)
        for p in i.projects:
            print(p)
        for r in i.researchers:
            print(r)
    
