# UZH-FS22-RL

### How to reproduce the result

a)	Clone the code repository from the link below
https://github.com/hk-kaden-kim/UZH-FS22-RL

-	Given the chess environment code was moved to /ChessEnvironment Folder.
-	RL_library.py includes the class and method for the agent training.
o	Adam
o	EpsilonGreedy_Policy
o	CalQvalues
o	BackProp
o	PerformanceCheck
o	EWM

b)	To train the chess agent, use python scripts
o	SARSA: RunSarsa.py
o	Qlearning: RunQlearning.py

* the discount factor γ and the speed β can set as run parameters. If there are no input, the default value will set; gamma = 0.85, beta = 0.00005.

e.g., python RunSarsa.py –-g 0.85 –b 0.0005

c)	After all episodes (5,000) done, result files are saved automatically at the same file location.
o	{SARSA: S, Qlearning: Q}_R_save
o	{SARSA: S, Qlearning: Q}_N_moves_save
o	{SARSA: S, Qlearning: Q}_FinalModel.pkl
-	This file save the neural network layer parameters (W, Bias) from the last episode run.

d)	The result file can be opened and analyzed through TrainingDataAnalysis.ipynb
