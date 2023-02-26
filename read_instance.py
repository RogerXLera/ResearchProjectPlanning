import csv
import os
from definitions import *

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


def read_project(file_,projects,researchers,targets,periods):
    """
        This function reads a project file
    """
    projects = [] 
    with open(file_,'r') as input_file:
        reader_ = csv.reader(input_file)
        id_ = 0
        for row in reader_:
            if row[0] == 'name':
                name_ = row[1]
            elif row[0] == 'pi':
                pi_ = row[1]
    
    return projects


if __name__ == "__main__":

    print("---------------MAIN------------------")
    path_ = os.getcwd()
    directory_r = 'data'
    file_ = 'researchers.csv'
    file_path = os.path.join(path_,directory_r,file_)
    R = read_researcher(file_path)
    for r in R:
        print(r)