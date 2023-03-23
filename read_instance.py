"""
Roger Lera
2023/01/07
"""
import csv
import os
import numpy as np
from definitions import *

def identify_researcher(name_id,list_):
    """
        This function returns the researcher given a name or id_
    """
    if isinstance(name_id,int):
        for r in list_:
            if name_id == r.id:
                return r
    elif isinstance(name_id,str):
        for r in list_:
            if name_id == r.name:
                return r
    else:
        print(f"Name_id {name_id} is neither a name or the id of a researcher.")

    print(f"name_id {name_id} does not belong to any researcher.")
    return None


def read_researcher(file_):
    """
        This function reads the researchers file 
    """
    researchers = []
    with open(file_,'r') as input_file:
        reader_ = csv.reader(input_file)
        id_ = 0
        for row in reader_:
            name_ = row[0]
            cost_ = float(row[1])
            time_ = int(row[2])
            if int(row[3]) == 0:
                contract_ = False
            else:
                contract_ = True
            r = Researcher(id_,name_,cost_,time_,contract_)
            researchers.append(r)
            id_ += 1

    return researchers


def read_project(file_,projects,researchers):
    """
        This function reads a project file
    """
    if len(projects) == 0:
        last_project_id = 0
    else:
        last_project_id = projects[-1].id

    with open(file_,'r') as input_file:
        reader_ = csv.reader(input_file)
        id_ = 0
        wp_ = []
        target_ = []
        period_ = []
        researchers_ = []
        for row in reader_:
            if row[0] == 'name':
                name_ = row[1]
            elif row[0] == 'pi':
                pi_ = row[1]
            elif row[0] == 'budget':
                budget_ = row[1]
            elif row[0] == 'wp':
                wp_.append(row[1:])
            elif row[0] == 'periods':
                period_.append(row[1:])
            elif row[0] == 'target':
                target_.append(row[1:])
            else:
                print(f"Line {row} from file {file_} is not valid input type.")
        
        pi_o = identify_researcher(pi_,researchers)
        try:
            p = Project(id_=last_project_id+1,name=name_,pi=pi_o,budget=float(budget_))
        except:
            raise ValueError(f"Missing data for project in file {file_}")
        
        for w in wp_:
            sd_ = w[1].split('/')
            sd_o = Month(int(sd_[1]),int(sd_[0]))
            ed_ = w[2].split('/')
            ed_o = Month(int(ed_[1]),int(ed_[0]))
            workpackage = WorkPackage(id_=int(w[0]),project=p,start_month=sd_o,end_month=ed_o,dedication=int(w[3]))
            workpackage.add(p.wp)

        for pe in period_:
            sd_ = pe[1].split('/')
            sd_o = Month(int(sd_[1]),int(sd_[0]))
            ed_ = pe[2].split('/')
            ed_o = Month(int(ed_[1]),int(ed_[0]))
            pe_o = Period(id_=int(pe[0]),start=sd_o,end=ed_o)
            pe_o.add(p.period)

        for t in target_:
            r = identify_researcher(t[0],researchers)
            r.add(p.researchers)
            r.projects.append(p)
            i = 0
            for pe in p.period:
                i += 1
                t_o = Target(project=p,researcher=r,period=pe,value=float(t[i]))
                t_o.add(p.target)
            
        
    return p

def instance_projects(folder_,researchers):
    """
        This function reads all the projects and returns a list of projects
    """
    files = os.listdir(folder_)
    projects = []
    for f in files:
        file_path = os.path.join(folder_,f)
        p = read_project(file_path,projects,researchers)
        p.add(projects)
    
    return projects


if __name__ == "__main__":

    print("---------------MAIN------------------")
    path_ = os.getcwd()
    directory_ = 'data'
    file_ = 'researchers.csv'
    file_path = os.path.join(path_,directory_,file_)
    file_projects = os.path.join(path_,directory_,'projects')
    R = read_researcher(file_path)
    for r in R:
        print(r)

    P = instance_projects(file_projects,R)
    for p in P:
        print(p)
        for w in p.wp:
            print(w)
        for t in p.target:
            print(t)
        for r in p.researchers:
            print(r)
        for pe in p.period:
            print(pe)
        sd,ed = p.date()
        print(f"Starting month of the project, {p.name}: {sd}")
        print(f"Ending month of the project, {p.name}: {ed}")