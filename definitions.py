"""
Roger Lera
2023/01/07
"""

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

    def test_input(id_,name,cost,time,contract):
        
        if type(id_) != int:
            raise ValueError(f"Researcher Id ({id_}) must be an integer.")
        if type(name) != str:
            raise ValueError(f"Researcher name ({name}) must be a string.")
        if type(cost) != float and type(cost) != int:
            raise ValueError(f"Researcher cost ({cost}) must be a float or int.")
        if type(time) != float and type(time) != int:
            raise ValueError(f"Researcher time ({time}) must be a float or int.")
        if type(contract) != float and type(contract) != int:
            raise ValueError(f"Researcher contract ({contract}) must be a bool.")
        

    def __init__(self,id_,name,cost=50.0,time=130,contract=True):
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
        string_ += f"\n Cost: {self.cost:.2f} \t Time: {self.time}"
        return string_

    def __repr__(self):
        return self.id
    

class Projects:
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

    def test_input(id_,name,cost,time,contract):
        
        if type(id_) != int:
            raise ValueError(f"Researcher Id ({id_}) must be an integer.")
        if type(name) != str:
            raise ValueError(f"Researcher name ({name}) must be a string.")
        if type(cost) != float and type(cost) != int:
            raise ValueError(f"Researcher cost ({cost}) must be a float or int.")
        if type(time) != float and type(time) != int:
            raise ValueError(f"Researcher time ({time}) must be a float or int.")
        if type(contract) != float and type(contract) != int:
            raise ValueError(f"Researcher contract ({contract}) must be a bool.")
        

    def __init__(self,id_,name,pi=None,cost=50.0,time=130,contract=True):
        self.id = id_
        self.name = name
        self.pi = pi
        self.budget = budget
        self.wp = []
        self.period = []
        self.researchers = []
        self.target = []
        
    def __str__(self):
        if self.contract:
            string_ = f"Project: {self.name} \t ID: {self.id} \n"
        else:
            string_ = f"P.I.: {self.pi.name} \t ID: {self.id} \n"
        string_ += f"\n Cost: {self.cost:.2f} \t Time: {self.time}"
        return string_

    def __repr__(self):
        return self.id
    


class TimePeriod:
    """
    This class stores the information about time periods and its methods.
    """
    def check_error(self,value):
        if value != None:
            if type(value) != int:
                raise ValueError(f"Expected integer and got {type(value)} with value {value}")
                return False
            elif value < 0:
                raise ValueError(f"Expected and integer bigger than 0 and got {value}.")
                return False
            else:
                return True
        else:
            return True

    def check(self):
        """
            This function checks if the self has sense, if it makes sense, it completes the duration and returns True.
            Otherwise, it returns False and it corrects the info by changing the duration or the finish date
            If there is not complete info, it remains unchanged and returns True
        """
        if self.start_date != None and self.finish_date != None:
            if self.start_date > self.finish_date:
                if self.duration != None and self.duration > 0:
                    self.finish_date = self.start_date + self.duration - 1
                    return False
                else:
                    self.duration = 1
                    self.finish_date = self.start_date + self.duration - 1
                    return False
            else:
                if self.duration != None:
                    if self.duration == self.finish_date - self.start_date + 1:
                        return True
                    else:
                        self.duration = self.finish_date - self.start_date + 1
                        return False
                else:
                    self.duration = self.finish_date - self.start_date + 1
                    return True

        elif self.start_date != None and self.duration > 0:
            if self.finish_date != None:
                if self.finish_date != self.start_date + self.duration - 1:
                    self.finish_date = self.start_date + self.duration - 1
                    return False
                else:
                    return True
            else:
                self.finish_date = self.start_date + self.duration - 1
                return True

        elif self.finish_date != None and self.duration > 0:
            if self.start_date != None:
                if self.start_date != self.finish_date - self.duration + 1:
                    self.start_date = self.finish_date - self.duration + 1
                    return False
                else:
                    return True
            else:
                self.start_date = self.finish_date - self.duration + 1
                return True
        
        elif self.start_date != None:
            self.duration = 1
            self.finish_date = self.start_date + self.duration - 1
            return True

        elif self.finish_date != None:
            self.duration = 1
            self.start_date = self.finish_date - self.duration + 1
            return True

        elif self.duration > 0:
            self.start_date = 0
            self.finish_date = self.start_date + self.duration - 1
            return True

        else:
            self.start_date = 0
            self.duration = 1
            self.finish_date = self.start_date + self.duration - 1
            return True

    

    def __init__(self,id_,start_date=None,finish_date=None,duration=None,tau=None):
        if self.check_error(id_):
            self.id = id_ #start date (week)
        if self.check_error(start_date):
            self.start_date = start_date #start date (week)
        if self.check_error(finish_date):
            self.finish_date = finish_date #finish date in (week)
        if self.check_error(duration):
            self.duration = duration # duration in (week)
        if self.check_error(tau):
            self.tau = tau #time slots available in the period: tau <= time_slot

        self.check()

    def __str__(self):
        string_ = f"Period: {self.id} \n Start Date: {self.start_date} \n Finish Date: {self.finish_date} \n"
        string_ += f"Duration: {self.duration} \n Tau: {self.tau}"
        return string_


    def update(self,start_date=None,finish_date=None,duration=None):
        
        self.check() # we check all okey
        if start_date != None:
            if self.check_error(start_date):
                self.start_date = start_date
                self.finish_date = self.start_date + self.duration - 1
        
        if finish_date != None:
            if self.check_error(finish_date):
                if finish_date < self.start_date:
                    raise ValueError(f"Input finish date ({finish_date}) is earlier than current start date ({self.start_date})")
                    return False
                self.finish_date = finish_date
                self.duration = self.finish_date - self.start_date + 1

        if duration != None:
            if self.check_error(duration):
                if duration == 0:
                    raise ValueError(f"Input duration ({duration}) has to be bigger than 0.")
                    return False
                self.duration = duration
                self.finish_date = self.start_date + self.duration - 1

        return True

        
class TimePeriodSequence:
    """
    This class stores the information about time periods and its methods.
    """

    def __init__(self,id=0):
        self.id = id
        self.sequence = []

    def __str__(self):
        string_ = f"Time Period Sequence: {self.id} \n"
        if len(self.sequence) == 0:
            string_ += "\t None"
        else:
            for p in self.sequence:
                string_ += f"\t Period {p.id} \t Start: {p.start_date} \t Tau: {p.tau} \n"
        return string_

    def add_period(self,period):
        len_ = len(self.sequence)
        if len_ == 0:
            self.sequence.append(period)
        else:
            last_period = self.sequence[-1]
            if period.start_date > last_period.finish_date:
                self.sequence.append(period)

            else:
                for i in range(len_):
                    curr_ = self.sequence[i]
                    if curr_.finish_date >= period.start_date:
                        if curr_.start_date <= period.start_date:
                            raise ValueError(f"Input Period ({period.start_date}-{period.finish_date}) overlaps with period ({curr_.start_date}-{curr_.finish_date})")
                        else:
                            if curr_.start_date <= period.finish_date:
                                raise ValueError(f"Input Period ({period.start_date}-{period.finish_date}) overlaps with period ({curr_.start_date}-{curr_.finish_date})")
                            else:
                                self.sequence.insert(i,period)


class Job:
    """
    This class stores the information about Jobs and its methods.
    """

    def __init__(self,id,name,descriptor=None):
        self.id = id
        self.name = name
        self.descriptor = descriptor
        self.skills = [] #skills needed for obtaining the job

    def __str__(self):
        string_ = f"Job: {self.name} \n Descriptor: {self.descriptor} \n Skills: \n"
        if len(self.skills) == 0:
            string_ += "\t None"
        else:
            for s in self.skills:
                string_ += f"\t Skill: {s.name} \t Level: {s.level} \n"

        return string_


class UserPreferences:
    """
    This class stores the information about the users preferences.
    """

    def __init__(self,id,name,nperiods=10,dedication=1,target_job=None):
        self.id = id
        self.name = name
        self.nperiods = nperiods
        self.dedication = dedication
        self.target_job = target_job

    def __str__(self):
        string_ = f"User Preference: {self.name} \n Number of periods: {self.nperiods} \n"
        string_ += f"Dedication per period: {self.dedication} \n Target Job: {self.target_job.name}"
        return string_


if __name__ == '__main__':

    print("---------------------------------------------------------------")
    print("SKILL")
    x = Skill('plant',2)
    print(x)
    print(x.name)
    print(x.level)
    y = Skill('feed',3)

    print("---------------------------------------------------------------")
    print("ACTIVITY")
    a = Activity(0,"Plant tree",1,1)
    print(a)
    print(a.id)
    print(a.name)
    print(a.skills)
    x.add_skill(a.skills)
    y.add_skill(a.skills)
    print(a.skills)
    print(a.prerequisites)
    print(a.time)
    print(a.cost)

    print("---------------------------------------------------------------")
    print("PERIOD")
    p = TimePeriod(3,4,3)
    print(p)
    print(p.start_date)
    print(p.finish_date)
    print(p.duration)
    print(p.tau)
    p.tau = 5
    print(p.tau)

    print("---------------------------------------------------------------")
    print("JOB")
    j = Job(0,"Journalist")
    print(j.id)
    print(j.name)
    print(j.descriptor)
    print(j.skills)
