import pickle
import matplotlib.pyplot as plt
import torch

data = pickle.load(open('speed_comparison.p', 'rb'))
sns_cpu = data['snsCPU']
sns_gpu = data['snsGPU']
torch_cpu = data['torchCPU']
torch_gpu = data['torchGPU']
speed = torch.tensor([4,8,16,32,64,128])

plt.figure()
plt.plot(speed[:-1],sns_cpu[:-1]/5,label='SNS-Toolbox (CPU)')
plt.plot(speed[:-1],sns_gpu[:-1]/5,label='SNS-Toolbox (GPU)')
plt.plot(speed,torch_cpu/5,label='SNSTorch (CPU)')
plt.plot(speed,torch_gpu/5,label='SNSTorch (GPU)')
plt.legend()
plt.xlabel('Network Size')
plt.ylabel('Time per Step (s)')
plt.yscale('log')
plt.xscale('log',base=2)
plt.show()