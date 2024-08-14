#https://towardsdatascience.com/q-learning-algorithm-from-explanation-to-implementation-cdbeda2ea187
#https://aleksandarhaber.com/q-learning-in-python-with-tests-in-cart-pole-openai-gym-environment-reinforcement-learning-tutorial/

#https://www.youtube.com/watch?v=2u1REHeHMrg



import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt

RENDER_FINAL_EPOCHS = True

class Q_agent:
    def __init__(self, num_class = 0, alpha= 0, gamma= 0, epsilon= 0, epoch= 0):

        self.num_class=num_class #discretizzazione dell'input
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epoch = epoch
        
        self.q_matrix = np.zeros(shape=(num_class+1,num_class+1,num_class+1,num_class+1,2))
        
  
        
    
    def learn_CartPole(self, render_final_epoch = True):

        self.env = gym.make('CartPole-v1')
        current_state = self.env.reset()
        self.upper_bounds = self.env.observation_space.high
        self.lower_bounds = self.env.observation_space.low
        
        pos_intervals = np.linspace(-2.4,2.4, self.num_class) #da documentazione, termina se esce da questo range
        vel_intervals = np.linspace(self.lower_bounds[1],self.upper_bounds[1],num_class)
        angle_intervals = np.linspace(-0.2095,0.2095, self.num_class) #da documentazione, termina se esce da questo range
        angle_vel_intervals = np.linspace(self.lower_bounds[3],self.upper_bounds[3],num_class)
        
        self.intervals = []
        #self.intervals.extend([pos_intervals,vel_intervals,angle_intervals,angle_vel_intervals])
      
        self.intervals = [np.linspace(self.lower_bounds[i],self.upper_bounds[i],num_class) for i in range(self.env.observation_space.shape[0])] 
        print(self.intervals)
        
        
        
        h_rewards=[]

        action = random.choice([0,1])
        
        d_current_state = self.discretize_state(current_state[0])
        
        tot_reward=0
        
        n_epoch=0
        
        rerandable=render_final_epoch
                
        while(n_epoch<self.epoch):
            
            action = self.select_action(n_epoch,d_current_state)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            d_next_state= self.discretize_state(next_state)
            #self.q_matrix[d_next_state[0],d_next_state[1],d_next_state[2],d_next_state[3],action] #
            #print(np.max(self.q_matrix[d_next_state[0],d_next_state[1],d_next_state[2],d_next_state[3],:]))
            if terminated :
                #reward*=(-1)
                n_epoch+=1
                observation = self.env.reset()[0]
                if(n_epoch%100==0):
                    print(f"Epoch: {n_epoch} | epsilon: {self.epsilon} |  alpha: {self.alpha} | reward: {tot_reward}")
                h_rewards.append(tot_reward)
                
                d_next_state=self.discretize_state(observation)    
                tot_reward=0
                d_current_state=d_next_state
                if (n_epoch>12000):
                    self.alpha = max(self.alpha-0.000008, 0.00001)

                self.epsilon=max(self.epsilon-0.00005,0.0001)
                
            #QLearing   
            self.q_matrix[d_current_state[0],d_current_state[1],d_current_state[2],d_current_state[3],action] = (1-self.alpha)*self.q_matrix[d_current_state[0],d_current_state[1],d_current_state[2],d_current_state[3],action]+self.alpha*(tot_reward+self.gamma*np.max(self.q_matrix[d_next_state[0],d_next_state[1],d_next_state[2],d_next_state[3],:]))
            #SARSA
            #self.q_matrix[d_current_state[0],d_current_state[1],d_current_state[2],d_current_state[3],action] = (1-self.alpha)*self.q_matrix[d_current_state[0],d_current_state[1],d_current_state[2],d_current_state[3],action]+self.alpha*(tot_reward+self.gamma*self.q_matrix[d_next_state[0],d_next_state[1],d_next_state[2],d_next_state[3],self.select_action(n_epoch,d_next_state)])    
             
            tot_reward+=reward
            
            d_current_state=d_next_state
            
            
            if ((n_epoch>self.epoch-500)and(rerandable)):
                rerandable=False
                self.env.close()
                self.env = gym.make('CartPole-v1',render_mode='human')
                current_state = self.env.reset()
                d_current_state = self.discretize_state(current_state[0])
                
                
    
        self.env.close()
        return h_rewards
        

        
    def select_action(self, n_epoch, d_state):
        #per avere esplorazione iniziale
        if n_epoch<8000:
            return np.random.choice([0,1])
        
        #self.gamma*=0.999
        
        
        if np.random.rand() < self.epsilon:
            return random.choice([0,1])
        else:
            return np.argmax(self.q_matrix[d_state[0],d_state[1],d_state[2],d_state[3],:])
            
        
    
    def discretize_state(self, state):
        d_state = [np.digitize(state[i],self.intervals[i]) for i in range(len(state))]
        return d_state
  
    

num_class = 25
alpha = 0.1
gamma =  1
epsilon = 0.8
epoch = 17000

 
agent = Q_agent(num_class, alpha, gamma, epsilon,epoch )
h_rewards= agent.learn_CartPole(render_final_epoch=RENDER_FINAL_EPOCHS) 
h_rewards = h_rewards[10000:]

x = np.arange(len(h_rewards))

# Calcola la regressione lineare
coefficients = np.polyfit(x, h_rewards, 1)
polynomial = np.poly1d(coefficients)
y_fit = polynomial(x)

# Plot dei dati e della regressione lineare
plt.plot(x, h_rewards, label='Dati', color='blue')
plt.plot(x, y_fit, label='Regressione Lineare', color='red')

# Aggiungi etichette e titolo
plt.xlabel('N-Epoch (after 10k)')
plt.ylabel('Reward')
plt.title('Grafico con Regressione Lineare')

# Mostra la legenda
plt.legend()

# Mostra il grafico
plt.show()



#finite le epoch salvare la policy su un pickle