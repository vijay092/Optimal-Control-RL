"""
A collection of operations on unit quaternions
"""

import numpy as np
            

def normalize(q):
    norm = np.sqrt(np.dot(q,q))
    return q/norm

def vec_to_uquat(v):
    dt = np.array(v).dtype
    if dt == np.int or dt == np.float:
        dt = float
    else:
        dt = object

    input_array = np.zeros(4,dtype=dt)
    input_array[1:] = v
    return input_array

def cast_as_uquat(q):
    if len(q) == 4:
        return q
    elif len(q) == 3:
        return vec_to_uquat(q)
        
def mult(q,p):
    qQuat = cast_as_uquat(q)
    pQuat = cast_as_uquat(p)

    if qQuat.dtype == np.object or pQuat.dtype == np.object:
        dt = object
    else:
        dt = float

    rQuat = np.zeros(4,dtype=dt)
        
    r1 = qQuat[0]
    v1 = qQuat[1:]
    r2 = pQuat[0]
    v2 = pQuat[1:]

    rQuat[0] = r1*r2 - np.dot(v1,v2)
    rQuat[1:] = r1*v2 + r2*v1 + np.cross(v1,v2)
    return rQuat
    
def inv(q):
    """
    This simply conjugates the quaternion.
    In contrast with a standard quaterionion, you would also need 
    to normalize. 
    """        
    if q.dtype == np.object:
        dt = object
    else:
        dt = float

    qinv = np.zeros(4,dtype=dt)
    # Here is the main difference between a unit quaternion
    # and a standard quaternion.
    qinv[0] = q[0] 
    qinv[1:] = -q[1:]
    return qinv
        
def rot(q,v):
    """
    This performs a rotation with a unit quaternion
    """
    qinv = inv(q)
    
    res = mult(q,mult(v,qinv))
    
    return np.array(res[1:])

def expq(v):
    """
    This computes the quaternion exponentiation of a vector, v
    
    Input v: a vector in R^3
    Output q: a unit quaternion corresponding to e^((0,v))
    """
    norm = np.sqrt(np.dot(v,v))
    qr = np.cos(norm)
    qv = v * np.sinc(norm/np.pi)
        
    # Want v * sin(norm) / norm. As is, you'll get problems in the limit
    # of norm near 0. Thus, use the sinc function, which is
    # sinc(x) = sin(pi*x) / (pi*x)

    q = np.array([qr,qv[0],qv[1],qv[2]])
    return q

def cross_mat(v):
    """
    Extract the skew symmetric matrix corresponding to a 3-vector
    """
    dt = v.dtype
    M = np.zeros((3,3),dtype=dt)
    
    M[0,1] = -v[2]
    M[0,2] = v[1]
    M[1,2] = -v[0]

    M = M - M.T
    return M

def mat(q):
    """
    return the rotation matrix corresponding to unit quaternion
    """
    dt = q.dtype
    r = q[0]
    v = q[1:]
    M = cross_mat(v)
    Msq = np.dot(M,M)
    R = r*r * np.eye(3) + np.outer(v,v) + 2 * r * M + Msq

    return R
