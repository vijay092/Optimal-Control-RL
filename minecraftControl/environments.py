import gym
import minecraftControl.minecraft as mc
import minecraftControl.controllers as ctrl
import numpy as np
import numpy.random as rnd
import minecraftControl.vehicles as vh

class minecraftEnvironment(gym.Env):
    def __init__(self,model,dt):

        self.dt = dt
        self.model = model 
        self.window = None
        self.reset_info()
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset_info(self):
        self.Done = False
        self.info = { 'InputFeasible' : True,
                      'StateFeasible' : True }
 

    def make_window(self):
        window = mc.Window(model=self.model,position=(0,3,0),flying=True,
                           height=800,width=800, caption='Pyglet',
                           resizable=True)


        self.window = window

    def step(self,action):

        if not self.action_space.contains(action):
            self.Done = True
            self.info['InputFeasible'] = False

        measurement = self.get_measurement()
        if not self.observation_space.contains(measurement):
            self.Done = True
            self.info['StateFeasible'] = False
            
        if not self.Done:
            # Check the input
            self.model.vehicle.update(self.dt,action)
        measurement = self.get_measurement()
        reward = self.model.vehicle.get_reward()

        if not self.observation_space.contains(measurement):
            self.Done = True
            self.info['StateFeasible'] = False
        
        return measurement,reward,self.Done,self.info
        

    def get_measurement(self):
        return self.model.vehicle.x
    
    def render(self):
        if self.window is None:
            self.make_window()
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.dispatch_event('on_draw')
        self.window.on_draw()
        self.window.flip()

    def close(self):
        if self.window is not None:
            self.window.close()
            self.window = None

    def initialize_state(self):
        # must override this
        self.x0 = np.zeros_like(self.model.vehicle.x)


    def reset(self):
        self.initialize_state()
        self.model.vehicle.set_state(self.x0)
        self.reset_info()
        return self.get_measurement()

        
class constrainedBall(minecraftEnvironment):
    def __init__(self):
        pos = np.zeros(2)
        vel = np.zeros(2)
        vehicle = vh.rollingSphere(pos,vel,.4,mc.VEHICLE_SPEED,controller=None)
        model = mc.Model(vehicle,mc.smallLayout) 

        self.action_space = gym.spaces.Box(low=-1.,high=1.,shape=(2,),dtype=np.float32)
        xHigh = np.array([2.5,2.5,.4,.4])
        self.observation_space = gym.spaces.Box(low=-xHigh,high=xHigh,dtype = np.float32)

        dt = 0.05
        super().__init__(model,dt)
            
    def initialize_state(self):
        
        pos = 5 * (rnd.rand(2) - .5)
        while np.max(np.abs(pos)) < 1.5:
            pos = 5 * (rnd.rand(2) - .5)
        vel = np.zeros(2)
        self.x0 = np.hstack([pos,vel])

class constrainedCar(minecraftEnvironment):
    def __init__(self):
        gain = 0.1
        vehicle = vh.car((0,-1,0),np.pi,1.,gain=gain,controller=None)
        model = mc.Model(vehicle,mc.smallLayout)

        self.gain = gain
        dt = 0.05

        
        uLow = np.array([-1,-1.])
        uHigh = np.array([1.,1.])
        self.action_space = gym.spaces.Box(low=uLow,high=uHigh,dtype = np.float32)

        xHigh = np.array([2.5,2.5,np.inf,2.,2.])
        xLow = np.array([-2.5,-2.5,-np.inf,-2.,-2.])
        self.observation_space = gym.spaces.Box(low=xLow,high=xHigh,dtype=np.float32)
        super().__init__(model,dt)
    
    def initialize_state(self):

        pos = 5 * (rnd.rand(2) - .5)
        while np.max(np.abs(pos)) < 1.5:
            pos = 5 * (rnd.rand(2) - .5)

        theta = 2*np.pi*rnd.rand()
        self.x0 = np.hstack([pos,theta,np.zeros(2)])

class constrainedQuadcopter(minecraftEnvironment):
    def __init__(self):
        dt = .05

        layout = mc.smallLayout
        vehicle = vh.quadcopter((0.,0,1),controller=None)
        model = mc.Model(vehicle,layout)
        
        super().__init__(model,dt)
