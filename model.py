import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        # basic nn with 2 hidden layers
        self.hidden1 = nn.Linear(28*28, 128)
        self.output = nn.Linear(128, 10)
    
    def forward(self, x):
        # __call__ of Linear class will call forward
        # i.e. self.hidden(x) == self.hidden.forward(x)
        y = torch.sigmoid(self.hidden1.forward(x))
        y = torch.sigmoid(self.output.forward(y))
        return y

def train(net, lr, epoch, x, y, batch_size):
    trainloader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    
    optimizer = optim.Adam(net.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    for e in range(epoch):
        print(f"Training epoch {e+1}")
        for i, data in enumerate(trainloader):
            inputs, targets = data
            optimizer.zero_grad()
            # convert np array to tensor, then flatten the array 
            # and convert it to float (required to be able to train)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # print(f'Item #{i}')
            # print(output)
            # print(target)
            # print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (e + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def test(net, x, y):
    correct = 0
    with torch.no_grad():
        for i in range(len(x)):
            _input = x[i]
            output = net(_input)
            prediction = torch.argmax(output)
            target = y[i]
            if prediction == target:
                correct += 1
    return correct

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    # convert from np arrays to tensors
    train_images = torch.tensor([image.flatten() for image in train_images]).float()
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_images = torch.tensor([image.flatten() for image in test_images]).float()
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    net = NeuralNet()
    print(net)
    train(net, 0.001, 3, train_images[:], train_labels[:], 4)

    num_correct = test(net, test_images, test_labels)
    print(f"Accuracy: {num_correct/len(test_labels) * 100}%")
