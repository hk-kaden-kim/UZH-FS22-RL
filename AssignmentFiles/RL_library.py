import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from generate_game import *
from Chess_env import *
from tqdm import tqdm

class Adam:

    def __init__(self, Params, beta1):
        
        N_dim=np.shape(np.shape(Params))[0] # It finds out if the parameters given are in a vector (N_dim=1) or a matrix (N_dim=2)
        
        # INITIALISATION OF THE MOMENTUMS
        if N_dim==1:
               
            self.N1=np.shape(Params)[0]
            
            self.mt=np.zeros([self.N1])
            self.vt=np.zeros([self.N1])
        
        if N_dim==2:
            
            self.N1=np.shape(Params)[0]
            self.N2=np.shape(Params)[1]
        
            self.mt=np.zeros([self.N1,self.N2])
            self.vt=np.zeros([self.N1,self.N2])
        
        # HYPERPARAMETERS OF ADAM
        self.beta1=beta1
        self.beta2=0.999
        
        self.epsilon=10**(-8)
        
        # COUNTER OF THE TRAINING PROCESS
        self.counter=0
        
        
    def Compute(self,Grads):
                
        self.counter=self.counter+1
        
        self.mt=self.beta1*self.mt+(1-self.beta1)*Grads
        
        self.vt=self.beta2*self.vt+(1-self.beta2)*Grads**2
        
        mt_n=self.mt/(1-self.beta1**self.counter)
        vt_n=self.vt/(1-self.beta2**self.counter)
        
        New_grads=mt_n/(np.sqrt(vt_n)+self.epsilon)
        
        return New_grads

def EpsilonGreedy_Policy(Qvalues, allowed_a, epsilon, verbose=False):
    
    a,_=np.where(allowed_a==1)
    allowed_Q=np.copy(Qvalues[a])

    rand_values=np.random.uniform(0,1)

    rand_a=rand_values<epsilon

    if verbose: print(rand_a)

    if rand_a==True:
        if verbose: print('random select')
        a_agent=np.random.permutation(a)[0]

    else:
        if verbose: print('greedy')
        a_i=np.argmax(allowed_Q)
        a_agent=np.copy(a[a_i])

    return (a_agent, a)

def CalQvalues(X, W1, W2, b1, b2):
    """
    Calculate Q Values of status X in Neural Network.
    This function assume that you have one hidden layer and
    using ReLu(Reactified Linear Function)as activation function.
    """

    # Frontpropagation: input layer -> hidden layer
    z_h = np.dot(W1,X)+b1
    a_h = z_h*(z_h>0)

    # Frontpropagation: hidden layer -> output layer
    z_output = np.dot(W2,a_h)+b2
    a_output = z_output*(z_output>0)

    Qvalues = a_output # 2D(1,N) result to 1D(N)

    return Qvalues, a_h

def PerformanceCheck(env,W1,W2,b1,b2,T):
    """
    performance check
    """
    
    S,X,allowed_a=env.Initialise_game()

    R_save = np.zeros([T, 1])
    N_moves_save = np.zeros([T, 1])
    max_move = 1000

    for n in tqdm(range(T)):

        S,X,allowed_a=env.Initialise_game()
        Done=0

        for i in range(max_move):

            Qvalues, A_h = CalQvalues(X,W1,W2,b1,b2) # Compute Q value depends on S (X:Features)
            a_agent,_ = EpsilonGreedy_Policy(Qvalues, allowed_a, 0) # Select the greedy action
            
            S,X,allowed_a,R,Done=env.OneStep(a_agent)
            
            if Done:
                R_save[n]=np.copy(R)
                N_moves_save[n]=np.copy(i)
                break

        if Done==0:
            R_save[n]=0
            N_moves_save[n]=max_move

    R_perf = np.mean(R_save)
    N_moves_perf = np.mean(N_moves_save)

    return R_perf, N_moves_perf



def EWM(Data,Episode,xlabel,ylabel,title,plot=False):
    data = pd.DataFrame(Data)
    ema = data.ewm(span=100, adjust=False).mean()
    
    if plot:
        # Comparison plot b/w stock values & EMA
        plt.scatter(Episode,list(data[0]), label="Data", s=1)
        plt.plot(Episode,list(ema[0]), label="EWM", color = 'r')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()
    return data,ema

