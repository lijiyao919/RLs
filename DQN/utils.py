import torch as T

# Device configuration
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print('The device is: ', device)

#Double DQN
DDQN = True

#Dueling DQN
DUELING = True