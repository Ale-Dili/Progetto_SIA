Il progetto è composto da 3 script python.


QLearning Cartpole:
Il modello si addestra al momento del runtime, si può scegliere di visualizzare 
il training nelle fasi finali attraverso la variabile RENDER_FINAL_EPOCHS


DQN Cartpole:
In questo script il modello è già addestrato (dqn_solver_v1), si può osservare
in azione tramite il parametro IS_TRAINED dello script DQN_cartpole.py. Impostandolo
a True, lo script caricherà il modello pre addestrato, a False ne addestrerà uno nuovo


A2C Breakout:
a2c_breakout.py -> scritp per addestrare il modello

A2C_breakout_7e6.zip -> modello addestrato

model_demonstration.py -> script per vedere il modello in azione.

