import numpy as np
from casadi import *

def collocation_tableau(c):
    """
    
    c, A, b = collocation_tableau(c)
    
    Computes the Butcher tableau corresponding to 
    a collection of collocation points, c
    
    Input 
    c - a collection of strictly increasing collocation points in [0,1]
    
    Returns
    c, A, b
    
    c - the original collocation points
    A - the tableau matrix
    b - the final weights
    
    """
    
    n = len(c) # length of the collocation points (1,2,3)

    x_plot = np.linspace(0,1,100) 
    Lag_polys = []
    Lag_ints = []
    b = []
    for i in range(n):
        # Construct the Lagrange Interpolating Polynomial
        not_c = np.hstack([c[:i],c[i+1:]]) # find c such that i =! j
        L = np.poly1d(not_c,r=True) # construct polynomial such that (x - c_1)(x - c_2).... 
        L = L / L(c[i]) # Top is the same poly and bottom is substituted with c_i
    
        L_i = L.integ() # Integrate to get a_ij without substitiuting
    
        # The ending weights
        b.append(L_i(1.)) # This is L_i evaluated at 1
    
        Lag_polys.append(L)
        Lag_ints.append(L_i)
    
    b = np.array(b)

    a = []
    for i in range(n):
        ci = c[i]
        row = []
        for j in range(n):
            L_j = Lag_ints[j]
            row.append(L_j(ci)) # evaluate at c_i
        
        
        a.append(row)
    a = np.array(a)
    return c,a,b 


def runge_kutta_equations(f,t,x,u,h,c,A,b):
    """
    G, K, x_next = runge_kutta_equations(f,x,h,c,A,b)
    
    Returns 
    G - a CasADi symbolic expression such that G == 0  
    corresponds to the Runge-Kutta equations.
    
    K - CasADi variables for the RK derivative evaluations
    
    x_next - A CasADi symbolic variable for the value of the 
    state at the end of the RK step. This is consistent with 
    multiple shooting. For single shooting, the variable x_next 
    could be eliminated . 
    
    Inputs 
    f - A function of the form x_dot = f(t,x) for the differential equation
    t - The initial time for evaluation
    x - An initial condition for the Runge-Kutta step
    h - A step size
    c,A,b - encodes a Butcher  Tableau
    """
    
    s = len(c)
    
    n = np.prod(x.shape)
    # First make sure that x is an explicit column vector
    x0 = x.reshape((n,1))
    u0 = u.reshape((1,1))
    
    
    K = MX.sym('K',(n,s)) # some symbol for K 
    
    # RK Evaluation points
    X_eval = x0 @ np.ones((1,s)) + h * K@A.T
    
    # The Time evaluation points
    t_eval = t + h *  np.array(c)
    u_eval = u0 + h *  np.array(c)
    
    G_list = []
    for i in range(s):
        G = K[:,i] - f(t_eval[i],X_eval[:,i],u_eval[i])
        G_list.append(G)
        
    # The value at the end of the step: This is the last equation
    x_next = MX.sym('x_next',(n,1))
    u_next = MX.sym('u_next')
    G = x_next - (x0 + h * K @ b.reshape((s,1)))
    G_list.append(G)
    G_list.append(x_next[0]/6 + u_next)
    shape_con = np.shape(x_next[0]/6 + u_next)
    G = vertcat(*G_list)
    
    
    
    return G,K,x_next,u_next,shape_con

flatten = lambda X : X.reshape((np.prod(X.shape),1))

def runge_kutta_sim(f,Time,x0,u0,c,A,b):
    """
    Simulate a differential equation via a Runge-Kutta Scheme
    """
    N = len(Time)-1
    
    x_c = DM(x0)
    u_c = DM(u0)
    
    G_list = []
    X_list = []
    K_list = []
    U_list = []

    for i in range(N):
        t = Time[i]
        h = Time[i+1]-Time[i]
        G,K,x_c,u_c,shape_con = runge_kutta_equations(f,t,x_c,u_c,h,c,A,b)
        G_list.append(G)
        K_list.append(K)
        X_list.append(x_c)
        U_list.append(u_c)

    
    # Stack the constraints
    G_all = vertcat(*G_list)
    UB = np.zeros(G_all.shape)
    sha = np.array(G_all.shape)
    shaa = sha - np.array([1,1])
    LB = np.zeros(shaa)
    
    # Stack the variables
    X_sym = horzcat(*X_list)
    K_sym = horzcat(*K_list)
    U_sym = horzcat(*U_list)
    

    X_flat = flatten(X_sym)
    K_flat = flatten(K_sym)
    U_flat = flatten(U_sym)
    Z = vertcat(X_flat,K_flat,U_flat)
    
    nlp = {'x' : Z, 'f' : 0, 'g' : G_all}


    opts = {'ipopt' : {'print_level' : 0},
            'print_time' : False,
            'error_on_fail': True}

    
    solver = nlpsol('solver','ipopt',nlp,opts)

    try:

        res = solver(ubg = UB,lbg = LB)
        X_RK = res['x'][:X_flat.shape[0],0].reshape(X_sym.shape)
        X_RK = horzcat(DM(x0),X_RK)
    
        X_RK = np.array(X_RK)
    
        return X_RK
    except RuntimeError:
        return None

