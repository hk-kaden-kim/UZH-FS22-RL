import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ChessEnvironment.degree_freedom_queen import *
from ChessEnvironment.degree_freedom_king1 import *
from ChessEnvironment.degree_freedom_king2 import *
from ChessEnvironment.generate_game import *
from ChessEnvironment.Chess_env import *
from tqdm import tqdm

class Adam:
    """
    This is the class for Adam optimization.
    Reference to the Lab 2 exercise.
    """
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
    """
    This is the class for the epsilon greedy policy.
    Reference to the Lab 2 exercise.
    """

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
    Calculate Q Values with the status X in Neural Network.
    This function assume that you have one hidden layer and
    using ReLu(Reactified Linear Function)as activation function.
    """

    # # Frontpropagation: input layer -> hidden layer
    z_h = np.dot(W1,X)+b1
    a_h = z_h*(z_h>0)

    # # Frontpropagation: hidden layer -> output layer
    z_output = np.dot(W2,a_h)+b2
    a_output = z_output*(z_output>0)

    Qvalues = a_output # 2D(1,N) result to 1D(N)

    return Qvalues, a_h

def BackProp(W2,W1,b2,b1,X,a_h,R,Q,A,AdamObj,gamma,eta,gameDone=False,Q_next=0,A_next=0):
    """
    This function is to update new parameter of the neural network
    by using Back Propagation.
    Depends on the status of chess game (Done or not),
    the error signal is computed in different way.
    """

    Qvalues = Q
    Qvalues_next = Q_next

    a_agent = A
    a_agent_next = A_next

    Adam_W2,Adam_W1,Adam_b2,Adam_b1 = AdamObj

    # Compute the error signal
    e_n = np.zeros(np.shape(Qvalues))
    if gameDone:
        e_n[a_agent] = R-Qvalues[a_agent]
    else:
        e_n[a_agent] = R+gamma*Qvalues_next[a_agent_next]-Qvalues[a_agent]

    # # Backpropagation: output layer -> hidden layer
    # # Activation Function : ReLu(Reactified Linear Function)
    delta2 = e_n
    dW2 = np.outer(delta2, a_h)
    db2 = delta2

    # # Backpropagation: hidden layer -> input layer
    # # Activation Function : ReLu(Reactified Linear Function)
    delta1 = np.dot(W2.T, delta2)
    dW1 = np.outer(delta1,X)
    db1 = delta1

    W2 += eta*Adam_W2.Compute(dW2)
    W1 += eta*Adam_W1.Compute(dW1)
    b2 += eta*Adam_b2.Compute(db2)
    b1 += eta*Adam_b1.Compute(db1)

    return W2,W1,b2,b1

def PerformanceCheck(env,W1,W2,b1,b2,T):
    """
    This function is for the performance check.
    Given the number of trial (T), the average reward and the number of moves per game
    are estimated.
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

def EWM(Data,Episode,xlabel,ylabel,title,color='b',plot=False):
    """
    This function is to plot the raw data with an exponential moving average line.
    """
    data = pd.DataFrame(Data)
    ema = data.ewm(span=100, adjust=False).mean()
    
    if plot:
        # Comparison plot b/w stock values & EMA
        plt.scatter(Episode,list(data[0]), label="Data", color = color, s=1, alpha = 0.5)
        plt.plot(Episode,list(ema[0]), label="EWM", color = color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()
    return data,ema