# Compute_Behaviour
Computation Model of Behavioural Data


The drift diffusion process for decision modelling and the reinforcement learning based models are implemented on MATLAB.
 The main.m file reproduces all the results and graphs in the guide document.
The computational models are analysed for performance and reliability.

Refer to the Guide for the overview and analysis of the project.  

Part 1 and Part 2 are two matlab files with data which model the behaviours through drift diffusion and reinforcement learning respectively.

# Data:

Each row represents a participant. First 23 rows are patients. Last 25 rows are
controls. Each column represents a trial. (Row number X represents participant
X in each file.)

choices.csv
    1 = choose stimulus A
    2 = choose stimulus B

rewards.csv
    0 = no reward
    1 = reward

