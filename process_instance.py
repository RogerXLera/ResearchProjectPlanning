"""
Roger Lera
2023/03/07
"""
from read_instance import read_researcher,instance_projects
from scipy.sparse import csr_matrix
from numba import jit
import numpy as np
import warnings
import time


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def planning_horizon(P):
    """
    This function will return the date of the start and the beginning of the
    planning horizon
    INPUT:
        :P(dict)
    RETURN: 
        list of the months keys
    """
    total_dur = project_duration(P)
    #print("|M| = ",total_dur)
    return list(range(1,total_dur+1))


def in_period(period,month):
    """
    This function checks if a month belongs into a period
    """
    if month >= period[0] and month <= period[1]:
        return True
    else:
        return False


def decision_variables(P,R,M):
    """
    This function computes the decision variables list Xp and Xw
    INPUT:
        :R (dict) researcher dictionary
        :P (dict) project dictionary
        :M (list) planning horizon
    RETURN: 
        xp (list) 
        xw (list)
    """
    xp = []
    xw = []
    
    for r in R.keys():
        proj_inv = list(R[r][2].keys())
        for p in proj_inv:
            p_date = [P[p][0],P[p][0]+P[p][1]-1] # [start,end]
            for w in P[p][4].keys():
                wp_date = [P[p][4][w][0],P[p][4][w][0]+P[p][4][w][1]-1] # [start,end]
                for m in M:
                    if in_period(wp_date,m) == True:
                        xw.append((r,(p,w),m))
            
            for m in M:
                if in_period(p_date,m) == True:
                    xp.append((r,p,m))

    #print("|Xp| = ",len(xp))
    #print("|Xw| = ",len(xw))
    return xp,xw


def target_vector(xp,R,P):
    """
    This function will return the target vector.
    INPUT:
        :xp (dict) decision variable
        :R (dict) researcher dictionary
        :P (dict) project dictionary
    RETURN: 
        target (np.array) 
    """
    
    target = [] # list containing all the values of the target vector
    
    for row in xp:
        # target hours per researcher
        target.append(R[row[0]][2][row[1]][row[2]])
    
    return np.array(target)


def dedication_vector(P):
    """
    This function will compute the d vector of the Formulation.
    INPUT:
        :P (dict)
    RETURN: d vector (np.array)
    """
    d_list = []
    
    for proj in P.keys():
        for w in P[proj][4].keys():
            d_list.append(P[proj][4][w][2])
   
    return np.array(d_list)


def tau_vector(R,M):
    """
    This function will compute the tau vector of the Formulation 5.0.
    INPUT:
        :R (dict)
        :M (list)
    RETURN: t (np.vector)
    """
    
    t = []
    for res in R.keys():
        for mon in M:
            t.append(R[res][1])

    return np.array(t)


def b_vector(P):
    """
    This function will compute the b vector of the Formulation.
    INPUT:
        :P (dict)
    RETURN: b (np.array)
    """
    
    b = []
    
    for proj in P.keys():
        b.append(P[proj][3])

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


def A_matrix(xp,xw):
    """
    This function will compute the A matrix of the Formulation.
    INPUT:
        :xp (list)
        :xw (list)
    RETURN: A matrix (np.matrix)
    """
    st = time.time()

    xp_p = [(r,p,m,0) for (r,p,m) in xp]
    xw_p = [(r,p,m,0) for (r,(p,w),m) in xw]
    vp = np.array(xp_p, dtype=np.uint16)
    vw = np.array(xw_p, dtype=np.uint16)
    A = np.equal(
        np.expand_dims(vp, axis=1).view(np.uint64),
        np.expand_dims(vw, axis=0).view(np.uint64)
    )
    A = np.squeeze(A)

    ft = time.time()
    #print(f"Time to compute A (dense): {ft-st}")
    return A


def D_matrix(P,xw):
    """
    This function will compute the D matrix of the Formulation.
    INPUT:
        :P (dict)
        :xw (list)
    RETURN: D matrix (np.matrix)
    """
    st = time.time()
    
    row_list = [(p,w) for p in P.keys() for w in P[p][4].keys()]
    col_list = [(p,w) for (r,(p,w),m) in xw]
    vp = np.array(row_list, dtype=np.uint16)
    vw = np.array(col_list, dtype=np.uint16)
    #print(f'Size: {sizeof_fmt(len(vp) * len(vw))}')
    D = np.equal(
        np.expand_dims(vp, axis=1).view(np.uint32),
        np.expand_dims(vw, axis=0).view(np.uint32)
    )
    D = np.squeeze(D)

    ft = time.time()
    #print(f"Time to compute D (dense): {ft-st}")

    return D


def T_matrix(R,M,xw):
    """
    This function will compute the T matrix of the Formulation.
    INPUT:
        :R (dict)
        :M (list)
        :xw (dict)
    RETURN: T matrix (np.matrix)
    """
    st = time.time()
    row_list = [(r,m) for r in R.keys() for m in M]
    col_list = [(r,m) for (r,(p,w),m) in xw]
    vp = np.array(row_list, dtype=np.uint16)
    vw = np.array(col_list, dtype=np.uint16)
    #print(f'Size: {sizeof_fmt(len(vp) * len(vw))}')
    T = np.equal(
        np.expand_dims(vp, axis=1).view(np.uint32),
        np.expand_dims(vw, axis=0).view(np.uint32)
    )
    T = np.squeeze(T)

    ft = time.time()
    #print(f"Time to compute T: {ft-st}")

    return T


def B_matrix(a, b, eur):
    st = time.time()
    B = np.equal(
        np.expand_dims(a, axis=1),
        np.expand_dims(b, axis=0)
    )
    B = np.multiply(
        B,
        np.expand_dims(eur, axis=0)
    )
    ft = time.time()
    #print(f"Time to compute B: {ft-st}")    
    return B
    

def B_matrix_old(P,R,xw):
    """
    This function will compute the C matrix of the Formulation.
    INPUT:
        :P (dict)
        :R (dict)
        :xw (dict)
    RETURN: B matrix (np.matrix)
    """
    st = time.time()
    B = []
    #running rows
    for proj in P.keys():
        #running cols
        B_row = []
        for col in xw:
            if proj==col[1][0]:
                B_row.append(R[col[0]][0])
            else:
                B_row.append(0)
        B.append(B_row)
    ft = time.time()
    #print(f"Time to compute B (old): {ft-st}")    
    return np.array(B)


def matrices(N_proj,wp_min,wp_max,dur_mean,dur_sd,ded_l,lambda_w,lambda_p,cost_min,cost_max,RS,costr_min,costr_max,av_min,av_max,rep=10,mu=1.0):

    # instance generator
    P,R = RPP_instance(N_proj,wp_min,wp_max,dur_mean,dur_sd,ded_l,lambda_w,lambda_p,cost_min,cost_max,RS,costr_min,costr_max,av_min,av_max,rep,mu)
    # compute planning horizon
    #print("Compute planning horizon")
    M = planning_horizon(P)
    # compute decision variables sets
    #print("Compute decision variables")
    xp,xw = decision_variables(P,R,M)

    # A matrix np.array of floats (|xp| x |xw|)
    print("Compute A")
    #A_dense = A_matrix(xp,xw)
    st = time.time()
    va = tuple2uint(
        [(r,p,m,0) for (r,p,m) in xp],
        np.uint16,
        np.uint64
    )
    vb = tuple2uint(
        [(r,p,m,0) for (r,(p,w),m) in xw],
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
    # np.testing.assert_array_equal(A.toarray(), A_dense)

    # D np.array of floats (|w| x |xw|)
    print("Compute D")
    #D_dense = D_matrix(P,xw)
    st = time.time()
    va = tuple2uint(
        [(p,w) for p in P.keys() for w in P[p][4].keys()],
        np.uint16,
        np.uint32
    )
    vb = tuple2uint(
        [(p,w) for (r,(p,w),m) in xw],
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
    # np.testing.assert_array_equal(D.toarray(), D_dense)

    # T np.array of floats (|R|·|M| x |xw|)
    print("Compute T")
    #T_dense = T_matrix(R,M,xw)
    st = time.time()
    va = tuple2uint(
        [(r,m) for r in R.keys() for m in M],
        np.uint16,
        np.uint32
    )
    vb = tuple2uint(
        [(r,m) for (r,(p,w),m) in xw],
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
    # np.testing.assert_array_equal(T.toarray(), T_dense)

    # target is t vector in the formulation (|xp| dim)
    #print("Compute target")
    t = target_vector(xp,R,P)

    # dedication of each work package (|w|)
    #print("Compute d")
    d = dedication_vector(P) 

    # tau vector (|R|·|M|) maximum hours per each month and staff researcher
    #print("Compute tau")
    tau = tau_vector(R,M)

    # B np.array of floats (|P| x |xw|)
    print("Compute B")
    st = time.time()
    va = np.array(list(P.keys()))
    vb = np.array([p for (r,(p,w),m) in xw])
    vr = np.array([R[r][0] for (r,(p,w),m) in xw])
    #B_dense = B_matrix(va, vb, vr)
    size, vac, vbc = count_equal(va, vb)
    _, B_R, B_C = coo(va, vb, size, vac, vbc)
    B_V = vr[B_C]
    B = csr_matrix((B_V, (B_R, B_C)), shape=(len(va), len(vb)))
    
    ft = time.time()
    print(f"Time to compute B (sparse): {ft-st}")
    print(f'Size of dense B: {sizeof_fmt(8 * len(va) * len(vb))}')
    print(f'Size of sparse B: {sizeof_fmt(B_V.nbytes + B_R.nbytes + B_C.nbytes)}')
    print(f'Density: {100 * len(B_V) / (len(va) * len(vb)):.2e}%')
    # np.testing.assert_array_equal(B.toarray(), B_dense)

    # b vector (|P|) budget per each project
    #print("Compute b")
    b = b_vector(P)

    return P,R,M,xp,xw,A,t,D,d,T,tau,B,b


def print_project_info(P):

    for p in P.keys():
        print(f"Project ID: {p} \t SD: {P[p][0]} \t Duration: {P[p][1]} \t Dedication: {P[p][2]} \t Budget: {P[p][3]:.2f}")
        for w in P[p][4].keys():
            print(f"\t WP ID: {w} \t SD: {P[p][4][w][0]} \t Duration: {P[p][4][w][1]} \t Dedication: {P[p][4][w][2]}")

    return None

def print_researcher_info(R,P):

    for r in R.keys():
        print(f"Researcher ID: {r} \t Cost: {R[r][0]:.2f} \t Availability: {R[r][1]}")
        for p in R[r][2].keys():
            #print(f"\t Project ID: {p}")
            for m in R[r][2][p].keys():
                #print(f"\t \t Month: {m} \t Target: {R[r][2][p][m]}")
                print(f"\t Project ID: {p} \t Target: {R[r][2][p][m]}")
                break

    return None

###############################################################################
################################## Test #######################################

if __name__ == '__main__':
    
    from instance_gen import RPP_instance,project_duration
    data = matrices(1000,1,25,38.07,4.70,0.000757,1000,0.179,26.35,44.33,0.39,17.16,54.01,120,140,rep=3)
    P,R,M,xp,xw,A,t,D,d,T,tau,B,b = data