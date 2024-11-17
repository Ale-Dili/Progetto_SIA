The project consists of 3 python scripts.


QLearning Cartpole:
The model trains at runtime, you can choose to display 
the training in the final stages through the RENDER_FINAL_EPOCHS variable.


DQN Cartpole:
In this script the model is already trained (dqn_solver_v1), it can be observed
in action through the IS_TRAINED parameter in the DQN_cartpole.py script. Setting it
to True, the script will load the pre-trained model, to False it will train a new one


A2C Breakout:
a2c_breakout.py -> script to train the model

A2C_breakout_7e6.zipper -> trained model

model_demonstration.py -> script to see the model in action.
