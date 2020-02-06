import numpy as np
from casadi import *
import runge_kutta as rk

def collocation_equations(f,x,u,t,h,c,A,b):
    """
    G,X,x_next = collocation_equations(f,t,x,h,c,A,b)
    
    Returns 
    G - a CasADi symbolic expression such that G == 0  
    corresponds to the Runge-Kutta equations.
    
    X - CasADi variables for the states at the collocation points 
    
    x_next - A CasADi symbolic variable for the value of the 
    state at the end of the RK step. This is consistent with 
    multiple shooting. For single shooting, the variable x_next 
    could be eliminated . 
    
    Inputs 
    f - A function of the form x_dot = f(t,x) for the differential equation
    t - The initial time for evaluation
    x - An initial condition for the Runge-Kutta step
    u - An input which is held constant over the step
    h - A step size
    c,A,b - encodes a Butcher  Tableau

    """
    
    n = np.prod(x.shape)
    x0 = x.reshape((n,1))

    s = len(c)
    
    X = MX.sym('X',(n,s))
    
    t_eval = t + h *  np.array(c)
    
    F_list = []
    for i in range(s):
        fi = f(t_eval[i],X[:,i],u)
        
        F_list.append(fi)
    F = horzcat(*F_list)
    
    G = x0 @  np.ones((1,s)) + h * F@A.T - X
    x_next = MX.sym('x_next',(n,1))
    
    G = vertcat(G.reshape((n*s,1)),x_next - x0 - h*F@b.reshape((s,1)))
    return G,X,x_next

def collocation_extra_constraints(g,X,u):
    """
    G,lb,ub = collocation_extra_constraints(g,X,u)
    
    Extra inequality constraints of the form 
    g(x,u) <= 0
    
    which are to be enforced at all of the collocation points.
    """
    
    m = X.shape[1]
    
    G_list = []
  
    for i in range(m):
        G_list.append(g(X[:,i],u))
        
    G = vertcat(*G_list)
    
    lb = -np.inf * np.ones(G.shape)
    ub = np.zeros(G.shape)
    return G,lb,ub

def collocation_optimize(f,g,Time,x0,nU,c):
    """
    
    """
    
    c,A,b = rk.collocation_tableau(c)
    
    U_sym = MX.sym('U',(nU,len(Time)-1))
    
    x_vars = []
    collocation_vars = []
    G_list = []
    ub_list = []
    lb_list = []
    for k in range(len(Time)-1):
        h = Time[k+1]-Time[k]
        t = Time[k]
    
        if k == 0:
            x_cur = np.copy(x0)
        else:
            x_cur = x_next
        
        u_cur = U_sym[:,k]
    
        G_col,X,x_next = collocation_equations(f,x_cur,u_cur,t,h,c,A,b)
    
        x_vars.append(x_next)
        collocation_vars.append(X.reshape((np.prod(X.shape),1)))
    
        G_col_flat = rk.flatten(G_col)
    
        G_list.append(G_col_flat)
        ub_list.append(np.zeros(G_col_flat.shape))
        lb_list.append(np.zeros(G_col_flat.shape))
    
        if k == 0:
            # We need to ensure that the first input is feasible
            X_all = horzcat(DM(x0),X,x_next)
        else:
            X_all = horzcat(X,x_next)
    
        G_con,lb_con,ub_con = collocation_extra_constraints(g,X_all,u_cur)
    
        G_list.append(G_con)
        ub_list.append(ub_con)
        lb_list.append(lb_con)
    
    G_all = vertcat(*G_list)
    ub_all = vertcat(*ub_list)
    lb_all = vertcat(*lb_list)
    
    X_traj = horzcat(*x_vars)
    Collocation_traj = vertcat(*collocation_vars)

    Z = vertcat(rk.flatten(X_traj),rk.flatten(U_sym),Collocation_traj)

    nlp = {'x' : Z, 'f' : X_traj[0,-1], 'g' : G_all}

    opts = {'ipopt' : {'print_level' : 0},
                'print_time' : False,
                'error_on_fail': True}

    
    solver = nlpsol('solver','ipopt',nlp,opts)
    
    try:
        res = solver(ubg = ub_all,lbg=lb_all)

        X_opt = res['x'][:np.prod(X_traj.shape)].reshape(X_traj.shape)
        X_opt = np.array(horzcat(x0,X_opt))
        
        U_opt = res['x'][np.prod(X_traj.shape):np.prod(X_traj.shape)+np.prod(U_sym.shape)]
        U_opt = np.array(U_opt).reshape(U_sym.shape)
        
        return X_opt,U_opt
    except RuntimeError:
        return None
