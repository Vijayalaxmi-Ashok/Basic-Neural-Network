#https://aladdinpersson.medium.com/pytorch-neural-network-tutorial-7e871d6be7c4
# https://aladdinpersson.medium.com/pytorch-neural-network-tutorial-7e871d6be7c4
#imports
import torch
import torch.nn as nn   #neura networkj has loss
import torch.optim as optim #all optimization
import torch.nn.functional as F  # all activations
from torch.utils.data import DataLoader  #dataset manage 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torchvision

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.0001
batch_size = 64
num_epoch = 1


import sys

class Identify(nn.Module):
    def __init__(self):
        super(Identify, self).__init__()


    def forward(self, x):
        return x

#load pretrained model & modify it
model = torchvision.models.vgg16(pretrained = True)
print(model)
model.avgpool = Identify()
model.classifier = nn.Linear(512, 10)
model.to(device)
print(model)



# sys.exit()

# load Data 
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# intialize the network 
model = NN(input_size = input_size, num_classes = num_classes).to(device)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# train network
for epoch in range(num_epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        #get data to cuda if possible
        data = data.to(device = device)
        targets = targets.to(device = device)

        #get to correct shape
        # print(data.shape)
        data = data.reshape(data.shape[0], -1)

        #forward
        scores = model(data)
        loss = criterion(scores, targets )

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam step
        optimizer.step()
 




# check accuracy on training and testing
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('checking accuracy onn training data')
    else:
        print('Checking accuracy on testing data')    
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        
        print(
            f"Got {num_correct} / {num_samples} with accuracy"
            f" {float(num_correct) / float(num_samples) * 100:.2f}"
        )
    
    model.train()
    # return acc

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)