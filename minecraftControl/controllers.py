import numpy as np
import scipy.linalg as la

   
class controller:
    """
    This is the base class for controllers

    The methods __init__, update, and value

    can be overridden for various applications
    """
    def __init__(self):
        """
        This initializes the strategy
        """

        # Each controller has a flag telling
        # when the task is completed
        
        self.Done = False
        
    def update(self,measurement,t):
        """
        The update method updates any internal state
        """
        pass

    def value(self):
        return None
        

class moveSphereTo:
    """
    controller = moveSphereTo(target_x,target_y,Kp,Kd,tolerance)

    target_x - desired x value
    target_y - desired y value
    Kp - optional proportional gain
    Kd - optional derivative gain
    tolerance - optional error tolerance to declare when goal is reached.
    """
    def __init__(self,target_x,target_y,Kp=3,Kd=7,tolerance=0.1):
        """
        This sets up the parameters
        """
        self.target = np.array([target_x,target_y])
        self.Kp = Kp
        self.Kd = Kd
        self.Done = False
        self.posErr = np.zeros(2)
        self.velErr = np.zeros(2)
        self.tolerance = tolerance
        
    def update(self,measurement,t):
        """

        """
        pos = measurement[:2]
        vel = measurement[2:]
        self.posErr = pos - self.target
        self.velErr = vel
        if la.norm(self.posErr) < self.tolerance:
            self.Done = True

            
    def value(self):
        """
        This computes the actions

        This actually returns the input
        """
        return -self.Kp * self.posErr - self.Kd * self.velErr

class controllerSequence:
    def __init__(self,controllers):
        self.NumControllers = len(controllers)
        self.controllers = controllers
        self.index = 0
        self.Done = False
        
    def update(self,measurement,t):
        if (self.controllers[self.index].Done) and (self.index < self.NumControllers -1):
            self.index += 1
        self.controllers[self.index].update(measurement,t)

        if (self.index == self.NumControllers - 1) and self.controllers[self.index].Done:
            self.Done = True
        

    def value(self):
        return self.controllers[self.index].value()
        
        
class turnCar:
    def __init__(self,target_angle,Kp=3,Kd=7,tolerance=0.05):
        self.target = target_angle
        self.Kp = Kp
        self.Kd = Kd
        self.tol = tolerance
        self.Done = False
        
    def update(self,measurement,t):
        x,y,theta,v,omega = measurement
        self.posError = ((theta - self.target + np.pi) % (2*np.pi)) - np.pi
        self.velError = omega

        if np.abs(self.posError) < self.tol and np.abs(self.velError) < self.tol:
            self.Done = True
        
    def value(self):
        domega = -self.Kp * self.posError - self.Kd * self.velError
        return np.array([0.,domega]) 

class carForward:
    def __init__(self,distance,Kp=3,Kd=7,Kp_ang=.01,Kd_ang=.01,tolerance=.1):
        # Distance Must be positive
        self.startPosition = None
        self.Kp = Kp
        self.Kd = Kd
        self.Kp_ang = Kp_ang
        self.Kd_ang = Kd_ang
        self.tol = tolerance
        self.Done = False
        self.goalPosition = None
        self.distance = np.abs(distance)
    def update(self,measurement,t):
        x,y,theta,v,omega = measurement
        curPos = np.array([x,y])
        if self.goalPosition is None:
            self.goalPosition = curPos + self.distance * np.array([np.cos(theta),np.sin(theta)])
            self.startPosition = np.copy(curPos)
            self.goalAngle = theta
            self.projector = (self.goalPosition - self.startPosition) / self.distance
        
        self.d_err = np.dot(curPos - self.startPosition,self.projector) -self.distance
        self.v_err = np.dot(np.array([v * np.cos(theta),v*np.sin(theta)]),self.projector)

        self.theta_err = ((theta - self.goalAngle + np.pi) % (2*np.pi)) - np.pi
        self.omega_err = omega

        if np.abs(self.d_err) < self.tol and np.abs(self.v_err) < self.tol:
            self.Done = True

    def value(self):
        return np.array([-self.Kp * self.d_err-self.Kd * self.v_err,-self.Kp_ang * self.theta_err - self.Kd_ang*self.omega_err]) 

class timedController(controller):
    def __init__(self,T):
        self.T = T
        self.ind = 0
        super().__init__()
    
    def update(self,measurement,t):
        
        # Find the first time index that is at least as large as the current 
        # time
        while (self.ind < len(self.T)-1) and (t >= self.T[min([self.ind+1,len(self.T)-1])]):
            self.ind += 1
                
        # If we've reached the end of our input sequence
        # Stop updating and mark as done
        if self.ind == len(self.T)-1:
            self.Done = True

        self.measurement = np.copy(measurement)
    
class timedOpenLoopSequence(timedController):
    """
    Apply a sequence of actions U a list of times T
    The sequence is applied in a sample-and-hold manner
    
    The lists U and T should have the same length.
    """
    def __init__(self,U,T):
        # Update is always
        self.U = U
        super().__init__(T)
        
            
    def value(self):
        return self.U[self.ind]

class timedAffineSequence(timedController):
    def __init__(self,Gains,Vecs,T):
        self.Gains = Gains
        self.Vecs = Vecs
        super().__init__(T)

    def value(self):
        K = self.Gains[self.ind]
        v = self.Vecs[self.ind]
        x = self.measurement
        return K@x + v

class timedFeedbackSequence(timedController):
    """
    Apply a sequence of static feedback Laws

    F is a list of static feedback rules
    T is a list of times that they should be applied

    Each f in F should have the form:
    action = f(measurement)
    """
    def __init__(self,Feedbacks,T):
        self.Feedbacks = Feedbacks
        super().__init__(T)


    def value(self):
        return self.Feedbacks[self.ind](self.measurement)
        

       
