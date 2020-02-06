import numpy as np
import numpy.random as rnd
import scipy.linalg as la

def calculateReturns(R,gamma=1.):
    T = len(R)
    G = np.array(R)
    gamVec = gamma**np.arange(T)
    G = G * gamVec
    G = G[::-1]
    G = np.cumsum(G)
    G = G[::-1]
    G = G / gamVec
    
    return G

class mcAgent:
    def __init__(self,env,gamma=1.,epsilon=0.01):
        self.env = env
        
        self.epsilon = epsilon
        self.gamma = gamma
        self.reset()
        
    def reset(self):
        self.q = np.zeros((self.env.observation_space.n,self.env.action_space.n))
        self.counts = np.zeros_like(self.q)
        self.resetLists()
        
    def resetLists(self):
        self.S = []
        self.A = []
        self.R = []
        
        
    def action(self,s):
        if rnd.rand() < self.epsilon:
            a = self.env.action_space.sample()
        else:
            a = np.argmax(self.q[s,:])
        return a
    
    def update_q(self):
        G = calculateReturns(self.R,self.gamma)
        
        for s,a,g in zip(self.S,self.A,G):
            self.counts[s,a] += 1
            c = self.counts[s,a]
            self.q[s,a] = (1/c) * g + ((c-1)/c) * self.q[s,a]
            
    
    def update(self,s,a,r,s_next,done,info):
        self.S.append(s)
        self.A.append(a)
        self.R.append(r)
        if done:
            self.update_q()
            self.resetLists()
            

            
            
class piAgent(mcAgent):
    def __init__(self,env,gamma=1.,epsilon=0.01):
        super().__init__(env,gamma,epsilon)
        
        self.r_counts = np.zeros_like(self.q)
        self.P_counts = np.ones((env.observation_space.n,env.action_space.n,env.observation_space.n))
        self.r = np.zeros_like(self.r_counts)
        
        self.terminalStates = []
        self.modelFromCounts()
        self.policyIteration()
        
    def modelFromCounts(self):
        self.P = np.zeros_like(self.P_counts)
        for s in range(self.env.observation_space.n):
            for a in range(self.env.action_space.n):
                self.P[s,a] = self.P_counts[s,a] / (self.P_counts[s,a].sum())
                
        # Force self loops on the terminal states
        for s in self.terminalStates:
            for a in range(self.env.action_space.n):
                self.P[s,a] = np.zeros(self.env.observation_space.n)
                self.P[s,a,s] = 1
                
    def updateModel(self):
        for s,a,r,s_next in zip(self.S[:-1],self.A,self.R,self.S[1:]):
            self.P_counts[s,a,s_next] += 1
            self.r_counts[s,a] += 1
            
            c = self.r_counts[s,a]
            self.r[s,a] = (1/c) * r + ((c-1)/c) * self.r[s,a]
            
        # For zero reward on the terminal states
        for s in self.terminalStates:
            self.r[s] = np.zeros(self.env.action_space.n)
            
        self.modelFromCounts()
        
    def action(self,s):
        if rnd.rand() < self.epsilon:
            a = self.env.action_space.sample()
        else:
            a = rnd.choice(self.env.action_space.n,p=self.policy[s])
            
        return a
    
    def policyIteration(self):
        policy = np.ones((self.env.observation_space.n,self.env.action_space.n)) / self.env.action_space.n
        
        policyStable = False
        while not policyStable:
            P = np.einsum('ijk,ij->ik',self.P,policy)
            r_bar = np.einsum('ij,ij->i',policy,self.r)
        
            v = la.lstsq(np.eye(len(r_bar))-self.gamma * P,r_bar)[0]
        
            q = self.r + self.gamma * np.einsum('ijk,k->ij',self.P,v)
            self.q = q
            newPolicy = np.zeros_like(policy)
            for s  in range(len(v)):
                a = np.argmax(q[s])
                e = np.zeros(self.env.action_space.n)
                e[a] = 1.
                newPolicy[s] = e
            
            if la.norm(policy - newPolicy) < 1e-6:
                policyStable = True
            else:
                policy = newPolicy
                
            # Hack
            #policyStable = True
            #policy = newPolicy
                
        self.policy = policy
            
            
    def update(self,s,a,r,s_next,done,info):
        self.S.append(s)
        self.A.append(a)
        self.R.append(r)
        TimeLimitReached = False
        if 'TimeLimit.truncated' in info.keys():
            if info['TimeLimit.truncated'] == True:
                TimeLimitReached = True
        if done:
            if not TimeLimitReached:
                self.S.append(s_next)
                if s_next not in self.terminalStates:
                    self.terminalStates.append(s_next)
                self.updateModel()
                self.policyIteration()
            self.resetLists()

class qAgent:
    def __init__(self,env,gamma=1.,epsilon=0.01,alpha = 1e-3):
        self.env = env
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.terminalStates = []
    
        self.reset()
        
    def reset(self):
        self.q = np.zeros((self.env.observation_space.n,self.env.action_space.n))
        self.counts = np.ones_like(self.q)
        
       
        
    def action(self,s):
        if rnd.rand() < self.epsilon:
            a = self.env.action_space.sample()
        else:
            a = np.argmax(self.q[s,:])
        return a
    
    def update_q(self,s,a,r,s_next):

        Q_next = np.max(self.q[s_next,:])
        self.counts[s,a] = self.counts[s,a] + 1
        c = self.counts[s,a]
        alpha = self.alpha
        alpha = (100./(1000+c))**.7
        
        delta = r + self.gamma * Q_next - self.q[s,a]
        q_new = self.q[s,a] + alpha * delta 
        self.q[s,a] = q_new 
        
                 
    
    def update(self,s,a,r,s_next,done,info):
        TimeLimitReached = False
        if 'TimeLimit.truncated' in info.keys():
            if info['TimeLimit.truncated'] == True:
                TimeLimitReached = True
        #if done:
        #    if not TimeLimitReached:
        #        if s_next not in self.terminalStates:
        #            self.terminalStates.append(s_next)
        #            self.q[s_next] = np.zeros_like(self.q[s_next])

        
        self.update_q(s,a,r,s_next)

        

