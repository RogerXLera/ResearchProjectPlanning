"""
Roger Lera
2023/01/07
"""
import numpy as np

class Month:
    """
    This class stores the information stored in a month
    """
    months = ["January","February","March","April","May","June",
            "July","August","September","October","November","December"]

    def test_input(self,month,year):
        
        if type(month) != int:
            raise ValueError(f"Month ({month}) must be an int.")
        if month <= 0 or month > 12:
            raise ValueError(f"Month ({month}) must be between 1 and 12 included.")
        if type(year) != int:
            raise ValueError(f"Year ({year}) must be an int.")

    def __init__(self,month,year):
        self.test_input(month,year)
        self.id = year*100 + month
        self.month = month
        self.year = year
        

    def __str__(self):
        return f"{self.months[self.month - 1]} of {self.year}"

    def __repr__(self):
        return self.id

    def add(self,list_):
        """
        This method adds a Month object within an ordered month list within the month
        returns the position where the element was added
        """
        len_ = len(list_)
        for i in range(len_ - 1,-1,-1):
            m = list_[i]
            if self.id == m.id: #if it is the same month, we delete and we do not add the same element
                return i
            elif self.id > m.id: #if the element is posterior, we add it in the next position
                list_.insert(i+1,self)
                return i
        list_.insert(0,self) #if the month is the earliest, we add it in the beggining of the list
        return 0

class Researcher:
    """
    This class stores the information about researchers and their features:
        - id: self generated ID -- int
        - name: name -- str
        - cost: cost per hour (€/h) -- float
        - time: number of time slots available per month -- int
        - contract: boolean indicating whether the researcher is contracted or
                    is a tenure researcher. 
    """

    def test_input(self,id_,name,cost,time,contract):
        
        if type(id_) != int:
            raise ValueError(f"Researcher Id ({id_}) must be an integer.")
        if type(name) != str:
            raise ValueError(f"Researcher name ({name}) must be a string.")
        if type(cost) != float and type(cost) != int:
            raise ValueError(f"Researcher cost ({cost}) must be a float or int.")
        if type(time) != float and type(time) != int:
            raise ValueError(f"Researcher time ({time}) must be a float or int.")
        if type(contract) != bool:
            raise ValueError(f"Researcher contract ({contract}) must be a bool.")

        return None
        

    def __init__(self,id_,name,cost=50.0,time=130,contract=False):
        self.test_input(id_,name,cost,time,contract)
        self.id = id_
        self.name = name
        self.cost = cost
        self.time = time
        self.contract = contract

    def __str__(self):
        if self.contract:
            string_ = f"Researcher: {self.name} \t ID: {self.id} (contract)\n"
        else:
            string_ = f"Researcher: {self.name} \t ID: {self.id} \n"
        string_ += f"\t Cost: {self.cost:.2f} \t Time: {self.time}"
        return string_

    def __repr__(self):
        return self.id

    def add(self,list_):
        """
        This method adds a Researcher object within a unordered list
        """
        if self not in list_:
            list_.append(self)
        return None


class WorkPackage:
    """
    This class stores the information about workpackages and their features:
        - id: self generated ID -- int
        - project: project belonging -- Project 
        - name: Project+id -- str
        - start: Start Month -- Month
        - end: End Month -- Month
        - dedication: number of time slots required (h) -- float/int
    """

    def test_input(self,id_,project,start_month,end_month,dedication):
        
        if type(id_) != int:
            raise ValueError(f"Work Package Id ({id_}) must be an integer.")
        if isinstance(project,Project) == False:
            raise ValueError(f"Work Package project ({project}) must be a Project object.")
        if isinstance(start_month,Month) == False:
            raise ValueError(f"Work Package start_month ({start_month}) must be a Month object.")
        if isinstance(end_month,Month) == False:
            raise ValueError(f"Work Package end_month ({end_month}) must be a Month object.")
        if type(dedication) != float and type(dedication) != int:
            raise ValueError(f"Work Package dedication ({dedication}) must be a float or int.")
        

    def __init__(self,id_,project,start_month,end_month,dedication):
        self.test_input(id_,project,start_month,end_month,dedication)
        self.id = project.id*1000 + id_
        self.name = f"{project.name}/{id_}"
        self.project = project
        self.start = start_month
        self.end = end_month
        self.dedication = dedication
        
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.id

    def add(self,list_):
        """
        This method adds a Work Package object within a unordered list
        """
        if self not in list_:
            list_.append(self)
        return None

class Period:
    """
        This class stores the information period of a Project
    """
    
    def test_input(self,id_,start,end):
        
        if type(id_) != int:
            raise ValueError(f"Period ID ({id_}) must be an int.")
        if isinstance(start,Month) == False:
            raise ValueError(f"Period start month ({start}) must be a month.")
        if isinstance(end,Month) == False:
            raise ValueError(f"Period end month ({end}) must be a month.")

    def __init__(self,id_,start,end):
        
        self.test_input(id_,start,end)
        self.id = id_
        self.start = start
        self.end = end
        

    def __str__(self):
        string = f"ID: {self.id} Start: {str(self.start)}\n"
        string += f"\t End: {str(self.end)}"
        return string

    def __repr__(self):
        return self.id

    def check(self,month):
        """
            Checks whether a month is within a period
        """
        if self.start.id <= month.id and self.finish.id >= month.id:
            return True
        else:
            return False

    def update(self,month):

        if self.start.id > month.id:
            self.start = month
        elif self.end.id < month.id:
            self.end = month
        
        return None
            

    def add(self,list_):
        """
        This method adds a Period object within an ordered period list within the month
        Returns the position where the element was added
        """
        len_ = len(list_)
        for i in range(len_ - 1,-1,-1):
            p = list_[i]
            start = p.start
            end = p.end 
            if self.start.id == start.id and self.end.id == end.id: #if it is the same month, we add the period as well
                list_.insert(i+1,self)
                return i
            elif self.start.id == start.id and self.end.id > end.id: #if the element is posterior, we add it in the next position
                list_.insert(i+1,self)
                return i
            elif self.start.id > start.id: #if the element is posterior, we add it in the next position
                list_.insert(i+1,self)
                return i

        list_.insert(0,self) #if the month is the earliest, we add it in the beggining of the list
        return 0

class Project:
    """
    This class stores the information about projects and their features:
        - id: self generated ID -- int
        - name: name -- str
        - pi: id of the researcher -- Researcher
        - budget: budget of the project (€) -- float
        - wp: number of time slots available per month -- int
        - period: periods of time -- list
        - researchers: researchers involved in project -- list
        - target: target of each researcher -- list
    """

    def test_input(self,id_,name,pi,budget):
        
        if type(id_) != int:
            raise ValueError(f"Project Id ({id_}) must be an integer.")
        if type(name) != str:
            raise ValueError(f"Project name ({name}) must be a string.")
        if isinstance(pi,Researcher) == False:
            raise ValueError(f"Project P.I. ({pi}) must be a Researcher.")
        if type(budget) != float and type(budget) != int:
            raise ValueError(f"Project budget ({budget}) must be a float or int.")
        

    def __init__(self,id_,name,pi=None,budget=1000000.0):
        self.test_input(id_,name,pi,budget)
        self.id = id_
        self.name = name
        self.pi = pi
        self.budget = budget
        self.wp = []
        self.period = []
        self.researchers = []
        self.target = []
        
    def __str__(self):
        string_ = f"Project: {self.name} \t ID: {self.id} \n"
        string_ += f"\t P.I.: {self.pi.name} \t Budget: {self.budget:.2f}"
        return string_

    def __repr__(self):
        return self.id

    def add(self,list_):
        """
        This method adds a Project object within a unordered list
        """
        if self not in list_:
            list_.append(self)
        return None

    def date(self):
        """
            This method returns the starting and ending date months
            return: list(start date month, end date month)
        """
        date_ = [np.inf,0]
        sd = None
        ed = None
        for w in self.wp:
            if w.start.id < date_[0]:
                date_[0] = w.start.id
                sd = w.start
            if w.end.id > date_[1]:
                date_[1] = w.end.id
                ed = w.end

        return sd,ed

    

class Target:
    """
    This class stores the information about projects and their features:
        - id: self generated ID -- int
        - project: -- Project
        - researcher: -- Researcher
        - period: -- Period
        - value: [0,1] dedication of the researcher in a project during a period -- float
    """

    def test_input(self,project,researcher,period,value):
        
        if isinstance(project,Project) == False:
            raise ValueError(f"Target project ({project}) must be a Project.")
        if isinstance(researcher,Researcher) == False:
            raise ValueError(f"Target researcher ({researcher}) must be a Researcher.")
        if isinstance(period,Period) == False:
            raise ValueError(f"Target period ({period}) must be a Period.")
        if type(value) != float:
            raise ValueError(f"Target value ({value}) must be a float.")
        if value < 0 or value > 1:
            raise ValueError(f"Target value, t ({value}) must be 0 <= t <= 1.")
        

    def __init__(self,project,researcher,period,value=1.0):
        self.test_input(project,researcher,period,value)
        self.id = project.id*10000 + researcher.id*10 + period.id
        self.project = project
        self.researcher = researcher
        self.period = period
        self.value = value
        
    def __str__(self):
        string_ = f"Target: {self.id} \t Value: {self.value} \n"
        string_ += f"\t Project: {self.project.name} \t Researcher: {self.researcher.name} \t Period: {str(self.period)}"
        return string_

    def __repr__(self):
        return self.id

    def add(self,list_):
        """
        This method adds a Target object within a unordered list
        """
        if self not in list_:
            list_.append(self)
        return None

        
class PlanningHorizon:
    """
    This class stores the information about the planning horizon.
    """

        
    def test_input(self,id_,period):
        
        if isinstance(id_,int) == False:
            raise ValueError(f"Planning Horizon ID ({id_}) must be an int.")
        if isinstance(period,Period) == False:
            raise ValueError(f"Planning Horizon start month ({period}) must be a Period.")
        
        
    def init_sequence(self):
        
        n_months = 12
        iter_ = self.start.id
        iter_month = self.start.month
        iter_year = self.start.year
        max_iter = self.end.id

        while iter_ <= max_iter:
            month_ = Month(month=iter_month,year=iter_year)
            month_.add(self.sequence)
            if iter_month < n_months:
                iter_month += 1
                iter_ += 1
            else:
                iter_month = 1
                iter_year += 1
                iter_ = iter_year*100 + iter_month
                
        return None
            
    def __init__(self,period,id_=0):
        self.test_input(id_,period)
        self.id = id_
        self.start = period.start
        self.end = period.end
        self.sequence = []
        self.init_sequence()
        

    def __str__(self):
        string = f"ID: {self.id} Start: {str(self.start)}\n"
        string += f"\t End: {str(self.end)}"

    def __repr__(self):
        return self.id

    def check(self,month):
        """
            Checks whether a month is within a period
        """
        if self.start.id <= month.id and self.finish.id >= month.id:
            return True
        else:
            return False

    def update(self,month_list):

        for month in month_list:
            if self.start.id > month.id:
                self.start = month
                
            elif self.end.id < month.id:
                self.end = month
        
        self.init_sequence()
        
        return None


if __name__ == '__main__':

    print("---------------------------------------------------------------")
    s = Month(month=1,year=2022)
    print(s)
    e = Month(month=12,year=2022)
    print(str(e))
    r = Researcher(id_=0,name="Paco",cost=50,time=130)
    print(r)
    print(type(r))
    p = Project(id_=0,name="Logistar",pi=r)
    print(p)
    wp = WorkPackage(id_=1,project=p,start_month=s,end_month=e,dedication=100)
    print(wp)
    a = Month(month=1,year=2022)
    print(a.id == s.id)
    pe = Period(id_ = 0,start=s,end=e)
    print(pe)
    aa = Month(month=1,year=2024)
    pe.update(aa)
    print(str(pe))