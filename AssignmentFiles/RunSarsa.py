"""
Chess Assignment - Sarsa
Training Python Script
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from generate_game import *
from Chess_env import *
from RL_library import *
import click

size_board = 4

@click.command()
@click.option('--g', type=float, default = 0.85)
@click.option('--b', type=float, default = 0.00005)

def main():

    """
    INITIALISE
    """
    ## INITIALISE THE ENVIRONMENT
    env=Chess_Env(size_board)
    S,X,allowed_a=env.Initialise_game()

    ## INITIALISE THE PARAMETERS
    N_a=np.shape(allowed_a)[0]   # TOTAL NUMBER OF POSSIBLE ACTIONS
    N_in=np.shape(X)[0]    ## INPUT SIZE
    N_h=200                ## NUMBER OF HIDDEN NODES

    ## INITIALISE NEURAL NETWORK
    W1 = np.random.randn(N_h,N_in) * np.sqrt(1 / (N_in))
    b1 = np.zeros((N_h,))
    W2 = np.random.randn(N_a,N_h) * np.sqrt(1 / (N_h))
    b2 = np.zeros((N_a,))

    beta1=0.9
    Adam_W1 = Adam(W1, beta1)
    Adam_b1 = Adam(b1, beta1)
    Adam_W2 = Adam(W2, beta1)
    Adam_b2 = Adam(b2, beta1)

    ## INITIALISE HYPERPARAMETERS
    epsilon_0 = 0.2      # STARTING VALUE OF EPSILON FOR THE EPSILON-GREEDY POLICY
    beta = beta          # THE PARAMETER SETS HOW QUICKLY THE VALUE OF EPSILON IS DECAYING (SEE epsilon_f BELOW)
    gamma = g            # THE DISCOUNT FACTOR
    eta = 0.0035         # THE LEARNING RATE

    """ 
    TRAINING
    """
    N_episodes = 20 #100000 # THE NUMBER OF GAMES TO BE PLAYED 
    N_perfCheck = 10 #Check the performance every N_perfCheck episodes

    # SAVING VARIABLES
    Episode_save = np.zeros([N_episodes//N_perfCheck, 1])
    R_save = np.zeros([N_episodes//N_perfCheck, 1])
    N_moves_save = np.zeros([N_episodes//N_perfCheck, 1])
    FinalModel_save = {'W1':W1,'W2':W2,'b1':b1,'b2':b2}

    p = 0
    for n in range(N_episodes):

        epsilon_f = epsilon_0 / (1 + beta * n)   ## DECAYING EPSILON
        Done=0                                   ## SET DONE TO ZERO (BEGINNING OF THE EPISODE)
        i = 1                                    ## COUNTER FOR NUMBER OF ACTIONS
        
        """
        Sarsa Algorithm
        1. Initialize S, X, ALLOWED_A
        """
        S,X,allowed_a=env.Initialise_game()      ## INITIALISE GAME
        
        """
        Sarsa Algorithm
        2. CHOOSE A_AGENT FROM S USING POLICY DERIVED FROM Q (EPSILON-GREEDY)
        """
        Qvalues, a_h = CalQvalues(X,W1,W2,b1,b2)
        a_agent, _ = EpsilonGreedy_Policy(Qvalues, allowed_a, epsilon_f)
        
        """
        Q-Learning Algorithm
            Loop for each episode
        """
        while Done==0:                        ## START THE EPISODE
            """
            Sarsa Algorithm
            3. TAKE ACTION A_AGENT, OBSERVE R, S_NEXT
            """
            S_next,X_next,allowed_a_next,R,Done=env.OneStep(a_agent)
            
            """
            If the game is Done(Checkmate, Draw),
            Update the parameters of Neural Network lastly.
            """        
            if Done==1:

                # Back Propagation with ADAM Optimization
                # NOT CONSIDER X_NEXT

                # Compute the error signal
                e_n = np.zeros(np.shape(Qvalues))
                e_n[a_agent] = R-Qvalues[a_agent]

                # Backpropagation: output layer -> hidden layer
                # Activation Function : ReLu(Reactified Linear Function)
                delta2 = e_n
                dW2 = np.outer(delta2, a_h)
                db2 = delta2

                # Backpropagation: hidden layer -> input layer
                # Activation Function : ReLu(Reactified Linear Function)
                delta1 = np.dot(W2.T, delta2)
                dW1 = np.outer(delta1,X)
                db1 = delta1

                W2 += eta*Adam_W2.Compute(dW2)
                W1 += eta*Adam_W1.Compute(dW1)
                b2 += eta*Adam_b2.Compute(db2)
                b1 += eta*Adam_b1.Compute(db1)
                
                break
        
            # IF THE EPISODE IS NOT OVER...
            else:
                
                ## ONLY TO PUT SUMETHING
                PIPPO=1
            """
            Sarsa Algorithm
            4. CHOOSE A_AGENT_NEXT FROM S_NEXT USING POLICY DERIVED FROM Q (EPSILON-GREEDY)
            """
            Qvalues_next, a_h = CalQvalues(X_next,W1,W2,b1,b2)
            a_agent_next, _ = EpsilonGreedy_Policy(Qvalues_next, allowed_a_next, epsilon_f)

            """
            Sarsa Algorithm
            5. UPDATE Q VALUES
            """
            # Compute the delta of Sarsa
            e_n = np.zeros(np.shape(Qvalues))
            e_n[a_agent] = R+gamma*Qvalues_next[a_agent_next]-Qvalues[a_agent]
            
            # Backpropagation: output layer -> hidden layer
            # Activation Function : ReLu(Reactified Linear Function)
            delta2 = e_n
            dW2 = np.outer(delta2, a_h)
            db2 = delta2

            # Backpropagation: hidden layer -> input layer
            # Activation Function : ReLu(Reactified Linear Function)
            delta1 = np.dot(W2.T, delta2)
            dW1 = np.outer(delta1,X)
            db1 = delta1

            W2 += eta*Adam_W2.Compute(dW2)
            W1 += eta*Adam_W1.Compute(dW1)
            b2 += eta*Adam_b2.Compute(db2)
            b1 += eta*Adam_b1.Compute(db1)

            """
            Q-Learning Algorithm
            6. UPDATE S, X, A_AGENT
            """  
            S=np.copy(S_next)
            X=np.copy(X_next)
            Qvalues, a_h = CalQvalues(X,W1,W2,b1,b2)
            a_agent=a_agent_next

            i += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS

        if ((n+1)%N_perfCheck==0): # PERFORMANCE CHECK
            print(n+1,': Agent Performance Check, START')
            R_perf, N_moves_perf = PerformanceCheck(env,W1,W2,b1,b2,100)
            print('Agent Performance Check, Average reward:',R_perf,'Number of steps: ',N_moves_perf)
            print('-------------------------------------------------------------------')

            Episode_save[p] = n+1
            R_save[p] = np.copy(R_perf)
            N_moves_save[p] = np.copy(N_moves_perf)
            
            p += 1

    """
    FINAL RESULT
    """
    R_perf, N_moves_perf = PerformanceCheck(env,W1,W2,b1,b2,100)
    print('Agent Performance Check, Average reward:',R_perf,'Number of steps: ',N_moves_perf)


    FinalModel_save['W1']=W1
    FinalModel_save['W2']=W2
    FinalModel_save['b1']=b1
    FinalModel_save['b2']=b2

    """
    SAVE RESULT
    """
    np.savetxt("S_Episode",Episode_save)
    np.savetxt("S_R_save",R_save)
    np.savetxt("S_N_moves_save",N_moves_save)
    file = open("S_FinalModel.pkl","wb")
    pickle.dump(FinalModel_save, file)
    file.close()


if __name__ == "__main__":
    main()