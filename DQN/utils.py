import torch as T

# Device configuration
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
print('The device is: ', device)