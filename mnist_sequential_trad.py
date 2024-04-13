import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
)
test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor()
)
loaders = {'train': torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True,num_workers=1),
           'test': torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),}

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        hidden = F.relu(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def test(model, loaders):
    # Test the model
    model.eval()
    # next(model_toolbox.parameters()).device
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            images = images.to(device)
            outputs = torch.zeros([len(images), 10])
            for i in range(len(images)):
                hidden = model.initHidden().to(device)
                image = images[i,:,:,:].squeeze()
                for j in range(image.shape[0]):
                    # print('%i %i' % (i, j))
                    output, hidden = model(image[j, :], hidden)
                outputs[i,:] = output
            pred_y = torch.max(outputs, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
    return accuracy
def train(num_epochs, model, loaders, name, optimizer, test_interval, loss_func):
    model.train()

    # Train the model
    total_step = len(loaders['train'])
    losses = [0]
    accuracy = [0]
    test_result = 0
    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(loaders['train']):
            image, label = image.squeeze().to(device), label.to(device)
            hidden = model.initHidden().to(device)
            for j in range(image.shape[0]):
                # print('%i %i'%(i,j))
                output, hidden = model(image[j,:], hidden)

            loss = loss_func(output, label)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i + 1) % test_interval == 0:
                print('{} Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(name, epoch + 1, num_epochs, i + 1, total_step,
                                                                            loss.item()))
                test_result = test(model, loaders)
                print(name + ' Test Accuracy: ' + str(test_result))
            accuracy.append(test_result)
            losses.append(loss.item())

    return losses, accuracy

input_size = 28
hidden_size = 128
output_size = 10
model = RNN(input_size, hidden_size, output_size).to(device)

loss_func = nn.CrossEntropyLoss()
num_epochs = 10

optim = optim.Adam(model.parameters(), lr = 0.01)

test_interval = 100
loss, accuracy = train(num_epochs, model, loaders, 'RNN', optim, test_interval, loss_func)

data = {'Loss': loss, 'Accuracy': accuracy}
pickle.dump(data, open('row_MNIST_rnn.p', 'wb'))

plt.figure()
plt.title('RNN Row-wise MNIST')
plt.plot(loss, label='Loss')
plt.plot(accuracy, label='Accuracy')
plt.legend()

plt.show()
