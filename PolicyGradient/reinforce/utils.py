import torch as T

# Device configuration
#device = T.device('cuda' if T.cuda.is_available() else 'cpu')
device = T.device('cpu')
print('The device is: ', device)