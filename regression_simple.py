from snstorch import modules as m
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

class SimpleNet(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.shape = [1]
        self.device = device
        self.source = m.NonSpikingLayer(self.shape, device=device)
        self.syn = m.NonSpikingChemicalSynapseElementwise(device=device)
        self.dest = m.NonSpikingLayer(self.shape, device=device)
        self.setup()

    def forward(self, x, state_source, state_dest):
        source_to_dest = self.syn(state_source, state_dest)
        state_source = self.source(x, state_source)
        state_dest = self.dest(source_to_dest, state_dest)
        return state_source, state_dest

    def init(self, batch_size=None):
        if batch_size is None:
            state_source = torch.zeros(self.shape, device=self.device)# + self.params['rest'].unsqueeze(1).unsqueeze(1).expand(self.shape)
            state_dest = torch.zeros(self.shape, device=self.device)# + self.params['rest'].unsqueeze(1).unsqueeze(1).expand(self.shape)
        else:
            batch_shape = self.shape.copy()
            batch_shape.insert(0,batch_size)
            state_source = torch.zeros(batch_shape, device=self.device)# + self.params['rest'].unsqueeze(1).unsqueeze(1).expand(batch_shape)
            state_dest = torch.zeros(batch_shape, device=self.device)# + self.params['rest'].unsqueeze(1).unsqueeze(1).expand(batch_shape)
        return state_source, state_dest

    def setup(self):
        self.syn.setup()

N_STEPS = 100
BATCH_SIZE = 64
N_ITER = 1000
N_TRIES = 5

for t in range(N_TRIES):
    teacher = SimpleNet()
    teacher.setup()
    student = SimpleNet()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    loss_history = []
    loss = 10
    i = 0
    while i < N_ITER and loss > 0.0001:
        sample = torch.rand(BATCH_SIZE)
        teacher_source, teacher_dest = teacher.init(batch_size=BATCH_SIZE)
        student_source, student_dest = student.init(batch_size=BATCH_SIZE)
        student.setup()
        optimizer.zero_grad()
        teacher_data = torch.zeros([BATCH_SIZE, N_STEPS])
        student_data = torch.zeros([BATCH_SIZE, N_STEPS])
        for j in range(N_STEPS):
            teacher_source, teacher_dest = teacher(sample, teacher_source, teacher_dest)
            student_source, student_dest = student(sample, student_source, student_dest)
            teacher_data[:,j] = teacher_dest[:,0]
            student_data[:,j] = student_dest[:,0]
        loss = criterion(student_data, teacher_data)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.detach().item())
        print('Iteration %i/%i | Batch %i/%i | Loss: %.4f'%(t, N_TRIES, i+1, N_ITER, loss))
        i+=1

    # Demo:
    # state_source, state_dest = teacher.init()
    # stim = torch.rand(1)
    # num_steps = 100
    # data = torch.zeros([2,num_steps])
    #
    # for i in range(num_steps):
    #     state_source, state_dest = teacher(stim, state_source, state_dest)
    #     data[0,i] = state_source
    #     data[1,i] = state_dest
    #
    # plt.figure()
    # plt.plot(data[0,:].detach())
    # plt.plot(data[1,:].detach())

    teacher_source, teacher_dest = teacher.init()
    student_source, student_dest = student.init()
    student.setup()
    optimizer.zero_grad()
    stim = torch.rand(1)
    data = torch.zeros([4,N_STEPS])
    for i in range(N_STEPS):
        teacher_source, teacher_dest = teacher(stim, teacher_source, teacher_dest)
        student_source, student_dest = student(stim, student_source, student_dest)
        data[0,i] = teacher_source
        data[1,i] = student_source
        data[2,i] = teacher_dest
        data[3,i] = student_dest

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(loss_history)
    plt.subplot(3,1,2)
    plt.title('Source')
    plt.plot(data[0,:].detach())
    plt.plot(data[1,:].detach())
    plt.subplot(3,1,3)
    plt.title('Destination')
    plt.plot(data[2,:].detach())
    plt.plot(data[3,:].detach())
plt.show()