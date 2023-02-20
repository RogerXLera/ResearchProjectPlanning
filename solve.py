from definitions import *

if __name__ == '__main__':

    
    s = Month(month=1,year=2022)
    print(s)
    e = Month(month=12,year=2022)
    print(e)
    print(type(s))
    print(isinstance(s,Month))
    r = Researcher(id_=0,name="Paco",cost=50,time=130)
    print(r)
    print(type(r))
    p = Project(id_=0,name="Logistar",pi=r)
    print(p)
    
    print(type(p))
    wp = WorkPackage(id_=1,project=p,start_month=s,end_month=e,dedication=100)
    print(wp)
