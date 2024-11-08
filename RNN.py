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

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.0001
batch_size = 64
num_epoch = 2




# create RNNN


# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

#     def forward(self, x):
#         # h0 is now a 2D tensor with the shape (num_layers, batch_size, hidden_size)
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
#         # Forward propagate RNN
#         out, _ = self.rnn(x, h0)
        
#         # Reshape output to (batch_size, hidden_size * sequence_length)
#         out = out.reshape(out.shape[0], -1)
        
#         # Pass through the fully connected layer
#         out = self.fc(out)
#         return out

# class RNN_GRU(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(RNN_GRU, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

#     def forward(self, x):
#         # h0 is now a 2D tensor with the shape (num_layers, batch_size, hidden_size)
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
#         # Forward propagate RNN
#         out, _ = self.gru(x, h0)
        
#         # Reshape output to (batch_size, hidden_size * sequence_length)
#         out = out.reshape(out.shape[0], -1)
        
#         # Pass through the fully connected layer
#         out = self.fc(out)
#         return out
# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

#     def forward(self, x):
#         # h0 is now a 2D tensor with the shape (num_layers, batch_size, hidden_size)
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#         # Forward propagate RNN
#         out, _ = self.lstm(x, (h0, c0))
        
#         # Reshape output to (batch_size, hidden_size * sequence_length)
#         out = out.reshape(out.shape[0], -1)
        
#         # Pass through the fully connected layer
#         out = self.fc(out)
#         return out

class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, bidirectional = True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:,-1,:])
        return out
# load Data 
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# intialize the network 
# model = RNN_GRU(input_size, hidden_size, num_layers, num_classes).to(device)
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# train network
for epoch in range(num_epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        #get data to cuda if possible
        data = data.to(device = device).squeeze(1)
        targets = targets.to(device = device)

        #get to correct shape
        # print(data.shape)
        # data = data.reshape(data.shape[0], -1)
        data = data.permute(0, 2, 1)
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
            x = x.to(device = device).squeeze(1)
            y = y.to(device = device)
            # x = x.reshape(x.shape[0], -1)
            x = x.permute(0, 2, 1)

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