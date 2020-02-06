from casadi import *
import numpy as np

# Here is a generic method for constructing MPC controllers
def buildMPCController(nX,nU,N,f,g,g_lb,g_ub,ell,Vf,Vf_ub):
    """
    Parameters
    nX - State Dimension
    nU - Input Dimension
    N - horizon of the controller
    f - dynamics function
    g - step constraint function
    g_ub - step constraint upper bound
    g_lb - step constraint lower bound
    ell - step cost
    Vf - final cost, assumed to be positive definite
    Vf_ub - upper bound on final final cost 
    """
    
    # First we build a symbolic representation
    # of the optimiziation problem
    #
    # THe only thing we don't put in is the initial
    # condition, since that will change when 
    # the 
    
    X_sym = MX.sym('X',(nX,N+1))
    U_sym = MX.sym('U',(nU,N))
    
    x_end = X_sym[:,-1]
    objective = Vf(x_end)
    
    constraints = [Vf(x_end)]
    lbs = [0.]
    ubs = [Vf_ub]
    
    for i in range(N):
        x_sym = X_sym[:,i]
        u_sym = U_sym[:,i]
        
        objective += ell(x_sym,u_sym)
        
        constraints.append(g(x_sym,u_sym))
        ubs.append(g_ub)
        lbs.append(g_lb)
        
        x_next = X_sym[:,i+1]
        constraints.append(x_next-f(x_sym,u_sym))
        ubs.append(DM.zeros(nX))
        lbs.append(DM.zeros(nX))
        
    X_len = np.prod(X_sym.shape)
    X_flat = X_sym.reshape((X_len,1))

    U_len = np.prod(U_sym.shape)
    U_flat = U_sym.reshape((U_len,1))
    Z = vertcat(X_flat,U_flat)
    
    g_all = vertcat(*constraints)
    ubs_all = vertcat(*ubs)
    lbs_all = vertcat(*lbs)
    
    # These are options to pass to the solver
    # Mainly it supresses printing
    opts = {'ipopt' : {'print_level' : 0},
            'print_time' : False,
            'error_on_fail' : True}

    def mpc_control(x):
        """
        This is the actual MPC controller
        """
        
        # Append the initial condition constraint
        g_x = vertcat(g_all,X_sym[:,0])
        ub_x = vertcat(ubs_all,x)
        lb_x = vertcat(lbs_all,x)
    
    
        # build the solver
        nlp = {'x' : Z, 'f' : objective, 'g' : g_x }
        nlp_solver = nlpsol('solver','ipopt',nlp,opts)


        # try to solve it
        try:
            mpc_res = nlp_solver(ubg=ub_x,lbg=lb_x)
    
            X_sol = mpc_res['x'][:X_len,0].reshape(X_sym.shape)
            U_sol = mpc_res['x'][X_len:,0].reshape(U_sym.shape)
            u0 = U_sol[:,0]
            return np.array(u0).flatten()
        
        except RuntimeError:
            return None
        
    return mpc_control
