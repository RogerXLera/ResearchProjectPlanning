"""
Roger Lera
2023/03/07
"""
from read_instance import read_researcher,instance_projects
from definitions import *
from scipy.sparse import csr_matrix
from numba import jit
import numpy as np
import warnings
import time
import os


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def planning_horizon(P):
    """
    This function will create the planning horizon
    """
    date_ = [np.inf,0]
    sd_t = None
    ed_t = None
    for p in P:
        sd,ed = p.date()
        if sd.id < date_[0]:
            date_[0] = sd.id
            sd_t = sd
        if ed.id > date_[1]:
            date_[1] = sd.id
            ed_t = ed
    pe = Period(id_= 0,start=sd,end=ed)
    return PlanningHorizon(period=pe)


def in_period(period,month):
    """
    This function checks if a month belongs into a period
    """
    if month.id >= period.start.id and month.id <= period.end.id:
        return True
    else:
        return False


def decision_variables(P):
    """
    This function computes the decision variables list Xp and Xw
    INPUT:
        P: (list) of all projects
    RETURN: 
        xp (list) 
        xw (list)
    """
    xp = []
    xw = []

    for p in P:
        for r in p.researchers:
            # xw
            for w in p.wp:
                pw = Period(id_= 0,start=w.start,end=w.end)
                Mw = PlanningHorizon(period=pw)
                for m in Mw.sequence:
                    xw.append((r,w,m))
            #xp
            sd,ed = p.date()
            pp = Period(id_= 0,start=sd,end=ed)
            Mp = PlanningHorizon(period=pp)
            for m in Mp.sequence:
                xp.append((r,p,m))
    
    return xp,xw

def decision_variables_2(P,R):
    """
    This function computes the decision variables list Xp and Xw
    INPUT:
        P: (list) of all projects
    RETURN: 
        xp (list) 
        xw (list)
    """
    xp = []
    xw = []

    for r in R:
        for p in P:
            # xw
            if r in p.researchers:
                for w in p.wp:
                    pw = Period(id_= 0,start=w.start,end=w.end)
                    Mw = PlanningHorizon(period=pw)
                    for m in Mw.sequence:
                        xw.append((r,w,m))
                #xp
                sd,ed = p.date()
                pp = Period(id_= 0,start=sd,end=ed)
                Mp = PlanningHorizon(period=pp)
                for m in Mp.sequence:
                    xp.append((r,p,m))
    
    return xp,xw


def target_vector(xp):
    """
    This function will return the target vector.
    INPUT:
        :xp (list)
    RETURN: 
        target (np.array) 
    """
    
    target = [] # list containing all the values of the target vector
        
    for (r,p,m) in xp:
        # target hours per researcher
        for t in p.target:
            if t.researcher.id == r.id and in_period(t.period,m):
                target.append(t.value * r.time)
                break

    if len(xp) != len(target):
        raise ValueError(f"Target vector (|t| = {len(target)}) has not the same dimension as Xp (|Xp| = {len(xp)})")
    
    return np.array(target)


def dedication_vector(P):
    """
    This function will compute the d vector of the Formulation.
    INPUT:
        :P (list)
    RETURN: d vector (np.array)
    """
    d_list = []
    
    for p in P:
        for w in p.wp:
            d_list.append(w.dedication)
   
    return np.array(d_list)


def tau_vector(R,M):
    """
    This function will compute the tau vector of the Formalisation.
    INPUT:
        :R (list)
        :M (P object)
    RETURN: t (np.vector)
    """
    
    t = []
    for r in R:
        for m in M.sequence:
            t.append(r.time)

    return np.array(t)


def b_vector(P):
    """
    This function will compute the b vector of the Formulation.
    INPUT:
        :P (list)
    RETURN: b (np.array)
    """
    
    b = []
    
    for p in P:
        b.append(p.budget)

    return np.array(b)


def tuple2uint(t, in_type, out_type):
    result = np.array(t, dtype=in_type).view(out_type)
    return np.squeeze(result)


def count_equal(A, B):
    As, Ac = np.unique(A, return_counts=True)
    Bs, Bc = np.unique(B, return_counts=True)
    inter, Ai, Bi = np.intersect1d(
        As,
        Bs,
        assume_unique=True,
        return_indices=True
    )
    return np.sum(np.multiply(Ac[Ai], Bc[Bi])), Ac, Bc


@jit(nopython=True)
def coo(A, B, size, A_c, B_c):
    A_as = np.argsort(A)
    B_as = np.argsort(B)
    A_s = A[A_as]
    B_s = B[B_as]
    R = np.empty(size, dtype=np.uint32)
    C = np.empty(size, dtype=np.uint32)
    o = 0
    i = 0
    j = 0
    i_c = 0
    j_c = 0
    # print(f'A:      {A}')
    # print(f'B:      {B}')
    # print(f'A_s:    {A_s}')
    # print(f'B_s:    {B_s}')
    # print(f'A_as:   {A_as}')
    # print(f'B_as:   {B_as}')
    while i < len(A) and j < len(B):
        # print(f'A_s[{i}] = {A_s[i]}')
        # print(f'B_s[{j}] = {B_s[j]}')
        if A_s[i] == B_s[j]:
            for h in range(A_c[i_c]):
                for k in range(B_c[j_c]):
                    # print(f'Element in position {i+h} in A_s was in position {A_as[i+h]} in A')
                    # print(f'Element in position {j+k} in B_s was in position {B_as[j+k]} in B')
                    R[o] = A_as[i + h]
                    C[o] = B_as[j + k]
                    o += 1
            i += A_c[i_c]
            j += B_c[j_c]
            i_c += 1
            j_c += 1
        elif A_s[i] < B_s[j]:
            i += A_c[i_c]
            i_c += 1
        else:
            j += B_c[j_c]
            j_c += 1
    return np.ones(size), R, C


def A_matrix(xw,xp):
    """
    This function will compute the A matrix of the Formulation.
    INPUT:
        :xp (list)
        :xw (list)
    RETURN: A matrix (np.matrix)
    """
    print("Compute A")
    st = time.time()
    va = tuple2uint(
        [(r.id,p.id,m.id,0) for (r,p,m) in xp],
        np.uint16,
        np.uint64
    )
    vb = tuple2uint(
        [(r.id,w.project.id,m.id,0) for (r,w,m) in xw],
        np.uint16,
        np.uint64
    )
    size, vac, vbc = count_equal(va, vb)
    A_V, A_R, A_C = coo(va, vb, size, vac, vbc)
    A = csr_matrix((A_V, (A_R, A_C)), shape=(len(va), len(vb)))
    ft = time.time()
    print(f"Time to compute A (sparse): {ft-st}")
    print(f'Size of dense A: {sizeof_fmt(len(va) * len(vb))}')
    print(f'Size of sparse A: {sizeof_fmt(A_V.nbytes + A_R.nbytes + A_C.nbytes)}')
    print(f'Density: {100 * len(A_V) / (len(va) * len(vb)):.2e}%')
    
    return A

def A_matrix_dense(xw,xp):
    """
    This function will compute the A matrix of the Formulation.
    INPUT:
        :xp (list)
        :xw (list)
    RETURN: A matrix (np.matrix)
    """
    print("Compute A")
    st = time.time()
    va = tuple2uint(
        [(r.id,p.id,m.id,0) for (r,p,m) in xp],
        np.uint16,
        np.uint64
    )
    vb = tuple2uint(
        [(r.id,w.project.id,m.id,0) for (r,w,m) in xw],
        np.uint16,
        np.uint64
    )
    A = np.equal(
        np.expand_dims(va, axis=1).view(np.uint64),
        np.expand_dims(vb, axis=0).view(np.uint64)
    )
    A = np.squeeze(A)
    ft = time.time()
    print(f"Time to compute A (dense): {ft-st}")
    print(f'Size of theoretical dense A: {sizeof_fmt(len(va) * len(vb))}')
    print(f'Size of A: {sizeof_fmt(A.nbytes)}')
    
    return A


def A_matrix_slow(xw,xp):
    """
    This function will compute the A matrix of the Formulation.
    INPUT:
        :xp (list)
        :xw (list)
    RETURN: A matrix (np.matrix)
    """
    print("Compute A slow")
    st = time.time()
    A = []
    for (r1,p1,m1) in xp:
        A_row = []
        for (r2,w2,m2) in xw:
            if r1.id == r2.id and p1.id == w2.project.id and m1.id == m2.id:
                A_row.append(1)
            else:
                A_row.append(0)
        A.append(A_row)
    A = np.array(A)
    ft = time.time()

    print(f"Time to compute A (slow): {ft-st}")
    print(f'Size of A: {sizeof_fmt(A.nbytes)}')
    
    return A


def D_matrix(xw,P):
    """
    This function will compute the D matrix of the Formulation.
    INPUT:
        :P (list)
        :xw (list)
    RETURN: D matrix (np.matrix)
    """
    print("Compute D")
    st = time.time()
    va = tuple2uint(
        [(p.id,w.id) for p in P for w in p.wp],
        np.uint16,
        np.uint32
    )
    vb = tuple2uint(
        [(w.project.id,w.id) for (r,w,m) in xw],
        np.uint16,
        np.uint32
    )
    size, vac, vbc = count_equal(va, vb)
    D_V, D_R, D_C = coo(va, vb, size, vac, vbc)
    D = csr_matrix((D_V, (D_R, D_C)), shape=(len(va), len(vb)))
    ft = time.time()
    print(f"Time to compute D (sparse): {ft-st}")
    print(f'Size of dense D: {sizeof_fmt(len(va) * len(vb))}')
    print(f'Size of sparse D: {sizeof_fmt(D_V.nbytes + D_R.nbytes + D_C.nbytes)}')
    print(f'Density: {100 * len(D_V) / (len(va) * len(vb)):.2e}%')
    
    return D

def D_matrix_dense(xw,P):
    """
    This function will compute the D matrix of the Formulation.
    INPUT:
        :P (list)
        :xw (list)
    RETURN: D matrix (np.matrix)
    """
    print("Compute D")
    st = time.time()
    va = tuple2uint(
        [(p.id,w.id) for p in P for w in p.wp],
        np.uint16,
        np.uint32
    )
    vb = tuple2uint(
        [(w.project.id,w.id) for (r,w,m) in xw],
        np.uint16,
        np.uint32
    )
    D = np.equal(
        np.expand_dims(va, axis=1).view(np.uint32),
        np.expand_dims(vb, axis=0).view(np.uint32)
    )
    D = np.squeeze(D)
    ft = time.time()
    print(f"Time to compute D (dense): {ft-st}")
    print(f'Size of theoretical dense D: {sizeof_fmt(len(va) * len(vb))}')
    print(f'Size of D: {sizeof_fmt(D.nbytes)}')
    
    return D

def D_matrix_slow(xw,P):
    """
    This function will compute the D matrix of the Formulation.
    INPUT:
        :P (list)
        :xw (list)
    RETURN: D matrix (np.matrix)
    """
    print("Compute D slow")
    st = time.time()
    D = []
    for p1 in P:
        for w1 in p1.wp:
            D_row = []
            for (r2,w2,m2) in xw:
                if w1.id == w2.id:
                    D_row.append(1)
                else:
                    D_row.append(0)
            D.append(D_row)
    D = np.array(D)
    ft = time.time()

    print(f"Time to compute D (slow): {ft-st}")
    print(f'Size of D: {sizeof_fmt(D.nbytes)}')
    
    return D


def T_matrix(xw,R,M):
    """
    This function will compute the T matrix of the Formulation.
    INPUT:
        :R (list)
        :M (PlanningHorizon object)
        :xw (list)
    RETURN: T matrix (np.matrix)
    """
    print("Compute T")
    st = time.time()
    va = tuple2uint(
        [(r.id,m.id) for r in R for m in M.sequence],
        np.uint16,
        np.uint32
    )
    vb = tuple2uint(
        [(r.id,m.id) for (r,w,m) in xw],
        np.uint16,
        np.uint32
    )
    size, vac, vbc = count_equal(va, vb)
    T_V, T_R, T_C = coo(va, vb, size, vac, vbc)
    T = csr_matrix((T_V, (T_R, T_C)), shape=(len(va), len(vb)))
    ft = time.time()
    print(f"Time to compute T (sparse): {ft-st}")
    print(f'Size of dense T: {sizeof_fmt(len(va) * len(vb))}')
    print(f'Size of sparse T: {sizeof_fmt(T_V.nbytes + T_R.nbytes + T_C.nbytes)}')
    print(f'Density: {100 * len(T_V) / (len(va) * len(vb)):.2e}%')

    return T

def T_matrix_dense(xw,R,M):
    """
    This function will compute the T matrix of the Formulation.
    INPUT:
        :R (list)
        :M (PlanningHorizon object)
        :xw (list)
    RETURN: T matrix (np.matrix)
    """
    print("Compute T")
    st = time.time()
    va = tuple2uint(
        [(r.id,m.id) for r in R for m in M.sequence],
        np.uint16,
        np.uint32
    )
    vb = tuple2uint(
        [(r.id,m.id) for (r,w,m) in xw],
        np.uint16,
        np.uint32
    )
    T = np.equal(
        np.expand_dims(va, axis=1).view(np.uint32),
        np.expand_dims(vb, axis=0).view(np.uint32)
    )
    T = np.squeeze(T)
    ft = time.time()
    print(f"Time to compute T (dense): {ft-st}")
    print(f'Size of theoretical dense T: {sizeof_fmt(len(va) * len(vb))}')
    print(f'Size of T: {sizeof_fmt(T.nbytes)}')
    
    return T


def T_matrix_slow(xw,R,M):
    """
    This function will compute the T matrix of the Formulation.
    INPUT:
        :R (list)
        :M (PlanningHorizon object)
        :xw (list)
    RETURN: T matrix (np.matrix)
    """
    print("Compute T")
    st = time.time()
    T = []
    for r1 in R:
        for m1 in M.sequence:
            T_row = []
            for (r2,w2,m2) in xw:
                if r1.id == r2.id and m1.id == m2.id:
                    T_row.append(1)
                else:
                    T_row.append(0)
            T.append(T_row)
    T = np.array(T)
    ft = time.time()
    print(f"Time to compute T (slow): {ft-st}")
    print(f'Size of T: {sizeof_fmt(T.nbytes)}')
    
    return T



def B_matrix(xw, P):
    """
    This function will compute the B matrix of the Formulation.
    INPUT:
        :P (list)
        :xw (list)
    RETURN: B matrix (np.matrix)
    """
    print("Compute B")
    st = time.time()
    va = np.array([p.id for p in P])
    vb = np.array([w.project.id for (r,w,m) in xw])
    vr = np.array([r.cost for (r,w,m) in xw])
    size, vac, vbc = count_equal(va, vb)
    _, B_R, B_C = coo(va, vb, size, vac, vbc)
    B_V = vr[B_C]
    B = csr_matrix((B_V, (B_R, B_C)), shape=(len(va), len(vb)))
    
    ft = time.time()
    print(f"Time to compute B (sparse): {ft-st}")
    print(f'Size of dense B: {sizeof_fmt(8 * len(va) * len(vb))}')
    print(f'Size of sparse B: {sizeof_fmt(B_V.nbytes + B_R.nbytes + B_C.nbytes)}')
    print(f'Density: {100 * len(B_V) / (len(va) * len(vb)):.2e}%')
    
    return B

def B_matrix_dense(xw, P):
    """
    This function will compute the B matrix of the Formulation.
    INPUT:
        :P (list)
        :xw (list)
    RETURN: B matrix (np.matrix)
    """
    print("Compute B")
    st = time.time()
    va = np.array([p.id for p in P])
    vb = np.array([w.project.id for (r,w,m) in xw])
    vr = np.array([r.cost for (r,w,m) in xw])
    B = np.equal(
        np.expand_dims(va, axis=1),
        np.expand_dims(vb, axis=0)
    )
    B = np.multiply(
        B,
        np.expand_dims(vr, axis=0)
    )
    
    ft = time.time()
    print(f"Time to compute B (dense): {ft-st}")
    print(f'Size of theoretical dense B: {sizeof_fmt(64 * len(va) * len(vb))}')
    print(f'Size of B: {sizeof_fmt(B.nbytes)}')
    
    return B

def B_matrix_slow(xw, P):
    """
    This function will compute the B matrix of the Formulation.
    INPUT:
        :P (list)
        :xw (list)
    RETURN: B matrix (np.matrix)
    """
    print("Compute B")
    st = time.time()
    B = []
    for p1 in P:
        B_row = []
        for (r2,w2,m2) in xw:
            if p1.id == w2.project.id:
                B_row.append(r2.cost)
            else:
                B_row.append(0.0)
        B.append(B_row)
    B = np.array(B)
    
    ft = time.time()
    print(f"Time to compute B (dense): {ft-st}")
    print(f'Size of B: {sizeof_fmt(B.nbytes)}')
    
    return B

def matrices(P,R):

    # compute planning horizon
    #print("Compute planning horizon")
    M = planning_horizon(P)
    # compute decision variables sets
    #print("Compute decision variables")
    xp1,xw1 = decision_variables(P)
    xp,xw = decision_variables_2(P,R)
    print("Length xp1: ",len(xp1))
    print("Length xw1: ",len(xw1))
    print("Length xp: ",len(xp))
    print("Length xw: ",len(xw))

    # A matrix np.array of floats (|xp| x |xw|)
    Aq = A_matrix(xw,xp)
    #A = A_matrix_dense(xw,xp)
    A = A_matrix_slow(xw,xp)
    print("Equal matrices? ",np.testing.assert_array_equal(Aq.toarray(), A))

    # D np.array of floats (|w| x |xw|)
    Dq = D_matrix(xw,P)
    #D = D_matrix_dense(xw,P)
    D = D_matrix_slow(xw,P)
    print("Equal matrices? ",np.testing.assert_array_equal(Dq.toarray(), D))

    # T np.array of floats (|R|·|M| x |xw|)
    Tq = T_matrix(xw,R,M)
    #T = T_matrix_dense(xw,R,M)
    T = T_matrix_slow(xw,R,M)
    print("Equal matrices? ",np.testing.assert_array_equal(Tq.toarray(), T))

    # target is t vector in the formulation (|xp| dim)
    #print("Compute target")
    
    t = target_vector(xp)
    """
    for i in range(len(t)):
        print(t[i])

    print(len(t))
    """
    # dedication of each work package (|w|)
    #print("Compute d")
    d = dedication_vector(P)
    for i in range(len(d)):
        print(d[i])

    print(len(d))


    # tau vector (|R|·|M|) maximum hours per each month and staff researcher
    #print("Compute tau")
    tau = tau_vector(R,M)
    """
    for i in range(len(tau)):
        print(tau[i])

    print(len(tau))
    """


    # B np.array of floats (|P| x |xw|)
    Bq = B_matrix(xw,P)
    #B = B_matrix_dense(xw,P)
    B = B_matrix_slow(xw,P)
    print("Equal matrices? ",np.testing.assert_array_equal(Bq.toarray(), B))

    # b np.array of floats (|P|)
    b = b_vector(P)
    """
    for i in range(len(t)):
        print(t[i])

    print(len(t))
    """

    return M,xp,xw,A,t,D,d,T,tau,B,b


###############################################################################
################################## Test #######################################

if __name__ == '__main__':
    
    print("---------------MAIN------------------")
    path_ = os.getcwd()
    directory_ = 'data'
    file_ = 'researchers.csv'
    file_path = os.path.join(path_,directory_,file_)
    file_projects = os.path.join(path_,directory_,'projects')
    R = read_researcher(file_path)
    P = instance_projects(file_projects,R)
    data = matrices(P,R)
    M,xp,xw,A,t,D,d,T,tau,B,b = data