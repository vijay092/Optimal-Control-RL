import numpy as np
import numpy.random as rnd
import scipy.linalg as la
import pyglet as pg
from pyglet.gl import *
from pyglet.window import key, mouse
import minecraftControl.controllers as ctrl

class vehicle:
    def __init__(self):
        self.Time = [0.]
        self.Done = False

    def on_key_press(self,symbol,modifiers):
        pass

    def update(self,dt,u=None):
        pass
    
    def draw(self):
        pass

    def on_key_release(self,symbol,modifiers):
        pass

    def get_camera_position(self):
        return np.zeros(3)

    def set_camera_position(self,x):
        pass

    def set_state(self,x):
        self.x = x

    def get_reward(self):
        return 0.

## Rolling Sphere ## 
numPrimes = 20
numMeridians = 20


VEHICLE_SPEED = 3


def sphere_vertices(x,y,z,n):

    Vertices = [] 
    top = np.array([x,y+n,z])

    Vertices.append(top)

    alpha = np.pi/(numPrimes+1)
    beta = 2*np.pi / (numMeridians)
    for p in range(numPrimes):
        theta = alpha * (p + 1)
        y_val = y+n*np.cos(theta)
        for m in range(numMeridians):
            phi = beta * m
            x_val = x + n* np.cos(phi) * np.sin(theta)
            z_val = z + n * np.sin(phi) * np.sin(theta)

            Vertices.append(np.array([x_val,y_val,z_val]))
            
            
    bottom = np.array([x,y-n,z])
    Vertices.append(bottom)

    Vertices = np.array(Vertices)
    return Vertices

def sphere_sequence():
    nv = numPrimes*numMeridians + 2
    Seq = []
    # Top triangles
    for p in range(numMeridians-1):
        Seq.extend([p+1,p+2,0])
    Seq.extend([numMeridians,1,0])

    # Middle Triangles
    # (1,2,5), (2,5,6),  (2,3,6), (3,6,7),  (3,4,7), (4,7,8),  (4,1,8), (1,8,5)
    # (5,6,9), (6,9,10), (6,7,10),(7,10,11),(7,8,11),(8,11,12),(8,5,12),(5,12,9)

    for p in range(numPrimes-1):
        offset = p * numMeridians
        for m in range(numMeridians-1):
            Seq.extend([offset+m+1,offset+m+2,offset+m+1+numMeridians])
            Seq.extend([offset+m+2,offset+m+1+numMeridians,offset+m+2+numMeridians])
        Seq.extend([offset+numMeridians,offset+1,offset+2*numMeridians])
        Seq.extend([offset+1,offset+2*numMeridians,offset+numMeridians+1])

    # 9 = 2*4+1 = (numPrimes-1) * numMeridians + 1
    # Bottom triangles
    # (13,9,10),(13,10,11),(13,11,12),(13,12,9)
    offset = (numPrimes-1) * numMeridians
    for p in range(numMeridians-1):
        Seq.extend([nv-1, offset + p+1, offset + p+2])
    Seq.extend([nv-1,offset+numMeridians,offset + 1])
    return Seq

    
sphereVertsColors = 60 * np.ones((2+numPrimes*numMeridians,3),dtype=int)
sphereVertsColors[:,2] = rnd.randint(0,256,size=len(sphereVertsColors))
sphereVertsColors = sphereVertsColors.flatten()

class rollingSphere(vehicle):
    def __init__(self,position,velocity,radius,SPEED,controller=None):
        self.R = np.eye(3)
        self.position = np.array(position)
        self.vertexColors = sphereVertsColors
        self.radius = radius
        self.Seq = sphere_sequence()
        # Assuming that y velocity is always zero
        # Just using a normalized velocity
        self.velocity = np.array(velocity)
        self.SPEED = SPEED
        self.MAXSPEED = 2 # Really the ratio to the max speed
        if controller is None:
            class nullController(ctrl.controller):
                def __init__(self):
                    super().__init__()
                def value(self):
                    return np.zeros(2)
            controller = nullController()
        self.controller = controller

        self.Time = [0.]
        self.x = np.hstack([self.position,self.velocity])
        self.Traj = [self.x]

        self.reward = -np.inf
        
        super().__init__()

    def worldToCameraPos(self,x):
        """
        Change coordinates from 2D world frame to 3D camera frame
        """
        return np.array([-x[0],-1,x[1]])

    def worldToCameraVel(self,v):
        return np.array([-v[0],0,v[1]])

    def get_camera_position(self):
        return self.worldToCameraPos(self.position)
    
    def set_camera_position(self,x):
        self.position = np.array([-x[0],x[2]])
    
    def get_vertices(self):
        camPos = self.worldToCameraPos(self.position)
        V = sphere_vertices(0,0,0,self.radius)
        VR = V@self.R.T
        
        return VR + np.outer(np.ones(len(V)),camPos)


    def get_angular_velocity(self):
        M = np.array([[0,0,-1],
                      [0,1,0],
                      [1,0,0]])

        v = self.worldToCameraVel(self.velocity)

        return M@v*self.SPEED /self.radius

    def update(self,dt,u=None):
        #x,y,z = self.position
        #sphereVertsColorsvx,vy,vz = self.velocity
        # Just controlling 2d, in more normal coordinates
        measurement = self.x
        t = self.Time[-1]
        if u is None:
            # The archicture assumes we first update the internal variables
            self.controller.update(measurement,t)
            self.Done = self.controller.Done
            # And then get the value
            dx,dz = self.controller.value()
            #print(dx,dy,dz)
            dy = 0.
            u = np.array([dx,dz])
        self.velocity += dt * u 
        s = la.norm(self.velocity)
        #if s > self.MAXSPEED:
        #    self.velocity = self.MAXSPEED * self.velocity / s
        self.position = self.position + dt * self.velocity * self.SPEED

        self.reward = -(la.norm(self.x)+la.norm(u))
        omega = self.get_angular_velocity()
        Omega = np.cross(omega,np.eye(3))

        self.R = la.expm(Omega * dt) @ self.R 

        self.x = np.hstack([self.position,self.velocity])
        self.Time.append(self.Time[-1]+dt)
        self.Traj.append(self.x)

    def set_state(self,x):
        self.position = x[:2]
        self.velocity = x[2:]
        self.x = np.copy(x)
        
    def get_reward(self):
        return self.reward
    
    def draw(self):
        Verts = self.get_vertices()
        Seq = self.Seq
        colors = self.vertexColors
        pg.graphics.draw_indexed(len(Verts),GL_TRIANGLES,
                                 Seq,
                                 ('v3f',Verts.flatten()),
                                 ('c3B',colors))


    def on_key_press(self,symbol,modifiers):
        if symbol == key.LEFT:
            self.velocity[0] -= 1
        elif symbol == key.RIGHT:
            self.velocity[0] += 1
        elif symbol == key.UP:
            self.velocity[1] += 1
        elif symbol == key.DOWN:
            self.velocity[1] -= 1

    def on_key_release(self,symbol,modifiers):
        if symbol == key.LEFT:
            self.velocity[0] += 1
        elif symbol == key.RIGHT:
            self.velocity[0] -= 1
        elif symbol == key.UP:
            self.velocity[1] -= 1
        elif symbol == key.DOWN:
            self.velocity[1] += 1

carBotY = -.1
carMidY = 0.075
carTopY = .15
cw = .08
cl = .08
carVertices = np.array([[cw*4.,carBotY,-4*cl],#0
                        [cw*4,carBotY,-2*cl],#1
                        [cw*4,carBotY,-1*cl],#2
                        [cw*4,carBotY,1*cl],#3
                        [cw*4,carBotY,2*cl],#4
                        [cw*4,carBotY,4*cl],#5
                        [cw*2,carBotY,5.5*cl],#6
                        [cw*-2,carBotY,5.5*cl],#7
                        [cw*-4,carBotY,4*cl],#8
                        [cw*-4,carBotY,2*cl],#9
                        [cw*-4,carBotY,1*cl],#10
                        [cw*-4,carBotY,-1*cl],#11
                        [cw*-4,carBotY,-2*cl],#12
                        [cw*-4,carBotY,-4*cl],#13
                        [cw*4.,carMidY,-4*cl],#14
                        [cw*4,carMidY,-2*cl],#15
                        [cw*3.,carTopY,-1*cl],#16
                        [cw*3.,carTopY,1*cl],#17
                        [cw*4,carMidY,2*cl],#18
                        [cw*4,carMidY,4*cl],#19
                        [cw*2,carMidY,5.*cl],#20
                        [cw*-2,carMidY,5.*cl],#21
                        [cw*-4,carMidY,4*cl],#22
                        [cw*-4,carMidY,2*cl],#23
                        [cw*-3.,carTopY,1*cl],
                        [cw*-3.,carTopY,-1*cl],
                        [cw*-4,carMidY,-2*cl],
                        [cw*-4,carMidY,-4*cl],
                        [cw*4,carMidY,-1*cl],
                        [cw*4,carMidY,1*cl],
                        [cw*-4,carMidY,-1*cl],
                        [cw*-4,carMidY,1*cl]])

carVertices = carVertices @ np.array([[0,0,-1],
                                      [0,1,0],
                                      [1,0,0]])

carSeq = [0,1,13,
          1,2,12,
          2,3,11,
          3,4,10,
          4,5,9,
          5,6,8,
          6,7,8,
          5,8,9,
          4,9,10,
          3,10,11,
          2,11,12,
          1,12,13,
          14,15,27,
          15,16,26,
          16,17,25,
          17,18,24,
          18,19,23,
          19,20,22,
          20,21,22,
          19,22,23,
          18,23,24,
          17,24,25,
          16,25,26,
          15,26,27,
          0,1,15,
          0,14,15,
          1,2,28,
          1,15,28,
          15,16,28,
          2,3,29,
          2,28,29,
          17,28,29,
          16,17,28,
          3,4,18,
          3,18,29,
          17,18,29,
          4,5,19,
          4,18,19,
          5,6,20,
          5,19,20,
          7,8,21,
          8,21,22,
          8,9,22,
          9,22,23,
          9,10,23,
          10,23,31,
          23,24,31,
          10,11,31,
          11,30,31,
          24,30,31,
          24,25,30,
          11,12,30,
          12,26,30,
          25,26,30,
          12,13,26,
          13,26,27,
          6,7,21,
          6,20,21,
          0,13,14,
          13,14,27]


carColors = np.array([[0,0,255],#0 
                      [0,0,255],#1
                      [0,0,255],#2
                      [0,0,255],#3
                      [0,0,255],#4
                      [0,0,255],#5
                      [0,0,255],#6
                      [0,0,255],#7
                      [0,0,255],#8
                      [0,0,255],#9
                      [0,0,255],#10
                      [0,0,255],#11
                      [0,0,255],#12
                      [0,0,255],#13
                      [35,45,155],#14
                      [35,45,155],#15
                      [155,100,70],#16
                      [155,100,70],#17
                      [235,45,55],#18
                      [235,45,55],#19
                      [70,50,155],#20
                      [70,50,155],#21
                      [235,45,55],#22
                      [235,45,55],#23
                      [155,100,70],#24
                      [155,100,70],#25
                      [35,45,155],#26
                      [35,45,155],#27
                      [0,0,0],#28
                      [0,0,0],#29
                      [0,0,0],#30
                      [0,0,0]]).flatten()                      
class car(vehicle):
    def __init__(self,position,orientation = np.pi,scale = 1.,
                 gain = 1,controller=None):
        self.gain = gain
        self.position = np.array(position).squeeze()
        self.theta = orientation
        self.scale = scale

        self.v = 0
        self.omega = 0.
        if controller is None:
            class nullController(ctrl.controller):
                def __init__(self):
                    super().__init__()
                def value(self):
                    return np.zeros(2)
            self.controller = nullController()
        else:
            self.controller = controller

        self.x = self.get_state()
        self.Time = [0.]
        self.ThetaTraj = [np.pi-orientation]
        self.XTraj = [-self.position[0]]
        self.YTraj = [self.position[2]]
        super().__init__()
    def get_rotation(self):
        theta = self.theta
        R = np.array([[np.cos(theta),0,-np.sin(theta)],
                      [0,1,0],
                      [np.sin(theta),0,np.cos(theta)]])
        return R

        
    def get_vertices(self): 
        V = carVertices
        nv = len(V)
        R = self.get_rotation()
        V = V @ R.T * self.scale + np.tile(self.position,(nv,1))
        return V 


    def draw(self):
        Verts = self.get_vertices() 
        Seq = carSeq
        colors = carColors
        pg.graphics.draw_indexed(len(Verts),GL_TRIANGLES,
                                 Seq,
                                 ('v3f',Verts.flatten()),
                                 ('c3B',colors))

    def set_state(self,X):
        self.position[0] = -X[0]
        self.position[2] = X[1]
        self.theta = np.pi - X[2]
        self.v = X[3]
        self.omega = -X[4]
        self.x = np.copy(X)
        
        
    def get_state(self):
        theta = self.theta
        x,_,y = self.position
        omega = self.omega
        v = self.v
        measurement = np.array([-x,y,np.pi-theta,v,-omega])
        return measurement

    def update(self,dt, u=None):
        #dpos = self.position_change(dt)
        
        theta = self.theta
        v = self.v

        if u is None:
            measurement = self.get_state()
            self.controller.update(measurement)
            u = self.controller.value()


        dv,domega = u

        self.v += self.gain * dv * dt 
        self.omega -= self.gain * domega * dt
        
        dpos = np.array([np.cos(theta),0,np.sin(theta)])*v*dt
 
        self.theta = self.theta + dt * self.omega 
        self.position = self.position + dpos


        self.x = self.get_state()
        self.Time.append(self.Time[-1]+dt)
        self.ThetaTraj.append(np.pi-self.theta)
        self.XTraj.append(-self.position[0])
        self.YTraj.append(self.position[2])
    
    def on_key_press(self,symbol,modifiers):
        if symbol == key.UP:
            self.v += 1
        elif symbol == key.DOWN:
            self.v -= 1
        elif symbol == key.LEFT:
            self.omega -= 1
        elif symbol == key.RIGHT:
            self.omega += 1

            
    def on_key_release(self,symbol,modifiers):
        if symbol == key.UP:
            self.v -= 1
        elif symbol == key.DOWN:
            self.v += 1
        elif symbol == key.LEFT:
            self.omega += 1
        elif symbol == key.RIGHT:
            self.omega -= 1



def cubeVertices(w,d,h):
    """
    These are cube vertices in the natural world frame
    """
    cubeVertices = np.array([[-w,-d,-h],
                             [-w,-d,h],
                             [-w,d,-h],
                             [-w,d,h],
                             [w,-d,-h],
                             [w,-d,h],
                             [w,d,-h],
                             [w,d,h]])
    return cubeVertices / 2.

cubeSequence = [0,1,2,
                1,2,3,
                0,1,4,
                1,4,5,
                0,2,4,
                2,4,6,
                1,3,5,
                3,5,7,
                2,3,6,
                3,6,7,
                4,5,6,
                5,6,7]

# coordinate transfromation from world frame to drawing frame
R_dw = np.array([[-1.,0,0],
                 [0,0,1],
                 [0,1,0]])
p_dw = np.array([0.,-1,0])

import minecraftControl.uquat as uq

class quadcopter(vehicle):
    def __init__(self,position,SPEED = VEHICLE_SPEED,controller=None):
        """
        In contrast to the earlier vehicles, which were coded in the drawing frame
        the position is in the more intuitive XYZ frame with:
        - X points east
        - Y points north
        - Z points up

        The transformation to the drawing frame is handled internally
        """
        super().__init__()
        self.SPEED = SPEED
        self.position = np.array(position).squeeze()
        
        self.bodyDimensions = [.12,.12,.07]
        self.bodyColors = rnd.randint(0,256,size=3*8)

        self.armDimensions = [.47,.02,.02]
        self.armColors = rnd.randint(0,50,size=3*8)

        self.bladeDimensions = [.13,0.01,.005]
        self.bladeColors = rnd.randint(100,150,size=3*8)
        
        self.bladeAngles = 2 * np.pi * rnd.rand(4)

        
        self.dynamicsParameters()

        M = np.diag([1,-1,1,-1])

        self.blade_flip = M
        
        self.blade_speed = M @ np.array([1,1,1,1]) * np.sqrt(self.M * self.g / (self.ct*4))


        # Body Velocity
        theta = 2*np.pi*rnd.rand()
        self.v = 0.05 * np.array([np.cos(theta),np.sin(theta),0])
        self.omega = np.array([0.,0.,0])
        
        # Body frame  to world frame rotation
        self.R = np.eye(3)

        self.v_traj = [self.v]
        self.Time = [0]
        
    def dynamicsParameters(self):
        """
        Parameters from robotics toolbox

        https://github.com/petercorke/robotics-toolbox-matlab
        """

        # Air density
        self.rho = 1.184

        self.g = 9.81
        # Mass
        self.M = 4.
        # Inertia
        J_flat =  np.array([0.082, 0.082, 0.149])
        self.J = np.diag(J_flat)
        self.J_inv = np.diag(1/J_flat)
        # Arm length
        self.d = 0.315
        # rotor height above CoG
        self.h = -0.007
        # Rotor Displacement Matrix
        self.D = np.array([[self.d,0,self.h],
                           [0,self.d,self.h],
                           [-self.d,0,self.h],
                           [0,-self.d,self.h]]).T

        # Rotor radius
        self.r = 0.165
        # Rotor disc area
        self.A = np.pi*self.r**2;
        # blade angles
        self.thetat = 6.8*(np.pi/180) 
        self.theta0 = 14.6*(np.pi/180)
        self.theta1 = self.thetat - self.theta0

        self.Mb = 0.005
        self.Mc = 0.010
        self.ec = 0.004
        self.Ib = self.Mb*(self.r-self.ec)**2/4
        self.Ic = self.Mc*(self.ec)**2/4; 
        
        self.a = 5.5
        self.c = 0.018
        self.gamma = self.rho*self.a*self.c*self.r**4/(self.Ib+self.Ic)
        
        # Thrust coefficient
        self.Ct = 0.0048
        # Torque coefficient
        self.Cq = self.Ct*np.sqrt(self.Ct/2)

    
       
        # Lumped thrust coefficient
        self.ct = self.Ct * self.rho * self.A * self.r**2
        # Lumped torque coefficient
        self.cq = self.Cq * self.rho * self.A * self.r**3

        # Torque-Thrust Transformation
        self.Gamma = np.array([[self.ct,self.ct,self.ct,self.ct],
                               [0,self.d*self.ct,0,-self.d*self.ct],
                               [-self.d*self.ct,0,self.d*self.ct,0],
                               [-self.cq,self.cq,-self.cq,self.cq]])
        
    def draw_indexed(self,Verts,Seq,colors):
        """
        Draws a sequence of vertices specified in the body frame.

        - Using self.R and self.position it first transforms to the world frame
        - Then, it uses R_dw and p_dw to transform to the drawing frame 
        """
        nv = len(Verts)
        # First transform to the world frame
        Verts = Verts @ self.R.T + np.tile(self.position,(nv,1))
        Verts = Verts @ R_dw.T + np.tile(p_dw,(nv,1))
        pg.graphics.draw_indexed(nv,GL_TRIANGLES,
                                 Seq,
                                 ('v3f',Verts.flatten()),
                                 ('c3B',colors))


    
    def update(self,dt, u=None):
        # Update the rotation matrix 

        # Calculate the rotational change
        # by Rodrigues formula
        
        theta = dt * la.norm(self.omega)
        if np.abs(theta) > 1e-12:
            w = self.omega / la.norm(self.omega)

            W = uq.cross_mat(w)
            W_sq = W @ W
        
            dR = np.eye(3) + np.sin(theta) * W + (1-np.cos(theta)) * W_sq

            # Then update the rotation
            self.R = self.R @ dR


        # Update the position
        # The somewhat more complex formula is because we are using the
        # body velocity, rather than the world-frame velocity
        self.position += dt * self.R @ self.v
        
        # Update the blades
        self.bladeAngles += dt * self.blade_speed

        # Calculate the forces and torques
        #F_gen = self.Gamma @ self.blade_speed**2
        #T_tot = F_gen[0]
        #F = np.array([0,0,T_tot])
        #tau = F_gen[1:]

        T = np.zeros((3,4))
        tau = np.zeros_like(T)
        Q = np.zeros_like(T)
        # More general dynamics involving blade flapping
        for i in range(4):
            Vr = np.cross(self.omega,self.D[:,i]) + self.v
            mu = la.norm(Vr) /(np.abs(self.blade_speed[i])*self.r)
            lc = Vr[2] / (np.abs(self.blade_speed[i])*self.r)
            psi = np.arctan2(Vr[1],Vr[0])
            J = np.array([[np.cos(psi),-np.sin(psi)],
                          [np.sin(psi),np.cos(psi)]])

            beta = np.array([((8/3*self.theta0 + 2*self.theta1)*mu - 2*(lc)*mu)/(1-mu**2/2),
                             0])
            beta = J.T @ beta
            a1s = beta[0] - 16 * self.omega[1] / (self.gamma * self.blade_speed[i])
            b1s = beta[1] - 16 * self.omega[0] / (self.gamma * self.blade_speed[i])

            # The ordering of this does not yet make sense to me.
            T[:,i] = self.ct * self.blade_speed[i]**2 * np.array([-np.cos(b1s)*np.sin(a1s),
                                                                  np.sin(b1s),
                                                                  np.cos(a1s)*np.cos(b1s)])

            tau[:,i] = np.cross(T[:,i],self.D[:,i])
            Q[:,i] = -self.cq*self.blade_speed[i]*np.abs(self.blade_speed[i]) * np.array([0,0,1]);
        # Update the velocities
        domega = self.J_inv@(-np.cross(self.omega,self.J@self.omega)+np.sum(tau+Q,axis=1))
        self.omega += dt * domega
        dv = (-np.cross(self.v,self.M * self.v) - self.M * self.g * self.R.T @np.array([0,0,1])+ np.sum(T,axis=1)) / self.M
        self.v += dt * dv

        self.Time.append(self.Time[-1] + dt)
        self.v_traj.append(self.v)

    def draw_body(self):
        V = cubeVertices(*self.bodyDimensions)
        Verts = V 
        Seq = cubeSequence
        self.draw_indexed(Verts,Seq,self.bodyColors)

                
    def draw_arms(self):
        V = cubeVertices(*self.armDimensions)
        Verts = V 
        Seq = cubeSequence
        Cols = self.armColors

        self.draw_indexed(Verts,Seq,Cols)

        R = np.array([[0,-1,0],
                      [1,0,0],
                      [0,0,1]])
        
        V = V @ R
        Verts = V 
        self.draw_indexed(Verts,Seq,self.armColors)

    def draw_blades(self):
        V = cubeVertices(*self.bladeDimensions)
        nv = len(V)
        w,d,h = self.armDimensions

        R = np.array([[0,-1,0],
                      [1,0,0],
                      [0,0,1]])
        

        armPosition = np.array([.9 * w/2,
                                0,
                                1.2 * d/2])


        
        Seq = cubeSequence
        Cols = self.bladeColors

        for i in range(4):
            if i > 0:
                armPosition = R @ armPosition 

            
            theta = self.bladeAngles[i]
            R_blade = np.array([[np.cos(theta),-np.sin(theta),0],
                                [np.sin(theta),np.cos(theta),0],
                                [0,0,1]])
            V_shift = V@R_blade + np.tile(armPosition,(nv,1))
            Verts = V_shift 
            self.draw_indexed(Verts,Seq,Cols)
         
        
    def draw(self):
        self.draw_body()
        self.draw_arms()
        self.draw_blades()                  
