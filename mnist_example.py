import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import snstorch.modules as m
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True,
)
test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor()
)

figure = plt.figure(figsize=(10, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

loaders = {'train' : torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True,num_workers=1),
                                                  'test'  : torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),}


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )  # fully connected layer, output 10 classes
        # self.out = nn.Linear(32 * 7 * 7, 10)
        self.out = nn.Sequential(nn.Flatten(), nn.LazyLinear(10))

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization


class SNSCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = m.NonSpikingChemicalSynapseConv(in_channels=1,
                                      out_channels=1,
                                      kernel_size=5,
                                      stride=1,
                                      padding=2)
        self.layer0 = m.NonSpikingLayer([28, 28])
        self.conv1 = m.NonSpikingChemicalSynapseConv(in_channels=1,
                                      out_channels=1,
                                      kernel_size=5,
                                      stride=1,
                                      padding=2)
        self.layer1 = m.NonSpikingLayer([28, 28])
        self.out = nn.Sequential(nn.Flatten(),
                                 m.PiecewiseActivation(),
                                 nn.LazyLinear(10))

    def forward(self, x):
        x = self.conv0(x, self.layer0.state_0)
        x = self.layer0(x)
        x = self.conv1(x, self.layer1.state_0)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # print(x.shape)
        x = self.layer1(x)
        output = self.out(x)
        return output, x  # return x for visualization

def test(model, loaders):
    # Test the model
    model.eval()
    # next(model.parameters()).device
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            images, labels = images.to(device), labels.to(device)
            test_output, last_layer = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass
    return accuracy
def train(num_epochs, model, loaders, name, optimizer):
    model.train()

    # Train the model
    total_step = len(loaders['train'])
    losses = [0]
    accuracy = [0]
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            images, labels = images.to(device), labels.to(device)

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)  # batch x
            b_y = Variable(labels)  # batch y
            output = model(b_x)[0]
            # if i == 0:
            #     print(b_y.shape)
            #     print(output.shape)
            loss = loss_func(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('{} Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(name, epoch + 1, num_epochs, i + 1, total_step,
                                                                            loss.item()))
                test_result = test(model, loaders)
                print(name + ' Test Accuracy: ' + str(test_result))
                accuracy.append(test_result)
            losses.append(loss.item())

    return losses, accuracy

cnn = CNN().to(device)
sns_cnn = SNSCNN().to(device)

loss_func = nn.CrossEntropyLoss()
num_epochs = 2

cnn_optim = optim.Adam(cnn.parameters(), lr = 0.01)
sns_optim = optim.Adam(sns_cnn.parameters(), lr = 0.01)

cnn_loss, cnn_accuracy = train(num_epochs, cnn, loaders, 'CNN', cnn_optim)
sns_loss, sns_accuracy = train(num_epochs, cnn, loaders, 'SNS', sns_optim)

data_cnn = {'Loss': cnn_loss, 'Accuracy': cnn_accuracy}
data_sns = {'Loss': sns_loss, 'Accuracy': sns_accuracy}
data = {'CNN': data_cnn, 'SNS': data_sns}
pickle.dump(data, open('MNIST_results.p', 'wb'))

plt.figure()
plt.subplot(1,2,1)
plt.title('Training Loss')
plt.plot(cnn_loss, label='CNN')
plt.plot(sns_loss, label='SNS')
plt.legend()
plt.subplot(1,2,2)
plt.title('Test Accuracy')
plt.plot(cnn_accuracy, label='CNN')
plt.plot(sns_accuracy, label='SNS')
plt.legend()

plt.show()
