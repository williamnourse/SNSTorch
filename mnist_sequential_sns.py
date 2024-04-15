import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import snstorch.modules as m
import pickle
from tqdm import tqdm
from datetime import datetime

BATCH_SIZE = 64

# list all transformations
transform = transforms.Compose(
    [transforms.ToTensor()])

# download and load training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# download and load testing dataset
testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# parameters
N_STEPS = 28
N_INPUTS = 28
N_NEURONS = 128
N_OUTPUTS = 10
N_EPOCHS = 10

class ImageSNS(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(ImageSNS, self).__init__()

        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.input = nn.Linear(self.n_inputs, self.n_neurons)
        self.hidden = m.NonSpikingLayer(self.n_neurons)
        self.h2h = m.NonSpikingChemicalSynapseLinear(self.n_neurons, self.n_neurons)
        # self.h2h = nn.Linear(self.n_neurons, self.n_neurons)
        self.out = nn.Linear(self.n_neurons, self.n_outputs)

    def init_hidden(self, size):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(size, self.n_neurons, device=device))

    def forward(self, X):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        X = X.permute(1, 0, 2)

        self.batch_size = X.size(1)

        # rnn_out => n_steps, batch_size, n_neurons (hidden states for each time step)
        # self.hidden => 1, batch_size, n_neurons (final state from each rnn_out)
        for i in range(N_STEPS):
            ext = self.input(X[i,:,:])
            prev = self.h2h(self.state, self.state)
            # prev = self.h2h(self.state)
            self.state = self.hidden(ext+prev, self.state)
            # print(ext+prev)
        out = self.out(self.state)
        # print(out)
        return out.view(-1, self.n_outputs)  # batch_size X n_output

device = 'cpu'
dataiter = iter(trainloader)
images, labels = next(dataiter)
# model = ImageSNS(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)
# logits = model(images.view(-1, 28,28))
# print(logits[0:10])

# Model instance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImageSNS(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

test_acc = 0.0
test_acc_last = -1.0
loss_history = []
acc_train_history = []
acc_test_history = []
# for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
epoch = 0
while test_acc > test_acc_last or epoch <= N_EPOCHS:
    train_running_loss = 0.0
    train_acc = 0.0
    model.train()

    # TRAINING ROUND
    for i, data in enumerate(tqdm(trainloader)):
        # print(i)
        # zero the parameter gradients
        optimizer.zero_grad()

        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(-1, 28, 28)

        batch_size = inputs.size(0)
        # reset hidden states
        model.state = model.init_hidden(batch_size)
        model.h2h.setup()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(outputs, labels, BATCH_SIZE)

    model.eval()
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f'
          % (epoch, train_running_loss / i, train_acc / i))
    loss_history.extend([train_running_loss/i])
    acc_train_history.extend([train_acc/i])
    test_acc_last = test_acc
    test_acc = 0.0
    model.h2h.setup()
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(-1, 28, 28)
        batch_size = inputs.size(0)
        model.state = model.init_hidden(batch_size)

        outputs = model(inputs)

        test_acc += get_accuracy(outputs, labels, BATCH_SIZE)
    acc_test_history.extend([test_acc/i])
    epoch += 1

    print('Test Accuracy: %.2f' % (test_acc / i))

save_data = {'loss': loss_history, 'accTrain': acc_train_history, 'accTest': acc_test_history}
pickle.dump(save_data, open('SNS-'+datetime.now().strftime('%d-%m-%Y-%H-%M-%S')+'.p', 'wb'))

plt.figure()
plt.subplot(2,1,1)
plt.plot(loss_history)
plt.title('Training Loss')
plt.subplot(2,1,2)
plt.plot(acc_train_history, label='Training Accuracy')
plt.plot(acc_test_history, label='Test Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()