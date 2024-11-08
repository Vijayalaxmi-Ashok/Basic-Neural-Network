# #https://aladdinpersson.medium.com/pytorch-neural-network-tutorial-7e871d6be7c4
# # https://aladdinpersson.medium.com/pytorch-neural-network-tutorial-7e871d6be7c4
# #imports
# import torch
# import torch.nn as nn   #neura networkj has loss
# import torch.optim as optim #all optimization
# import torch.nn.functional as F  # all activations
# from torch.utils.data import DataLoader  #dataset manage 
# import torchvision.datasets as datasets 
# import torchvision.transforms as transforms

# # Create fully connected network
# # class NN(nn.Module):
# #     def __init__(self, input_size, num_classes):
# #         super(NN,  self).__init__()
# #         self.fc1 = nn.Linear(input_size, 50)
# #         self.fc2 = nn.Linear(50, num_classes)

# #     def forward(self,  x):
# #         x = F.relu(self.fc1(x))
# #         x = self.fc2(x)
# #         return x 
    
            

# # Todo : create simple CNN
# class CNN(nn.Module):
#     def __init__(self, in_channels = 1, num_classes = 10):
#         super(CNN , self).__init__()
#         self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size  = (3,3), stride = (1, 1), padding = (1, 1))             
#         self.pool = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
#         self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
#         self.fc1 = nn.Linear(16*7*7, num_classes)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc1(x)

#         return x


# def save_checkpoint(state, filename = "my_checkpoint1.pth.tar"):
#     print("=> Saving checkpoint")
#     torch.save(state, filename)


# def load_checkpoint(checkpoint):
#     print("=> Loading checkpoint")
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])


# # model = CNN()
# # x = torch.randn(64, 784)
# # print(model(x).shape)

# # set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # hyperparameters
# input_size = 784
# num_classes = 10
# learning_rate = 0.0001
# batch_size = 64
# num_epoch = 1
# load_model = True

# # load Data 
# train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
# train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
# test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)
# test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# # intialize the network 
# # model = NN(input_size = input_size, num_classes = num_classes).to(device)
# model = CNN().to(device)

# # loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# if load_model:
#     load_checkpoint(torch.load("my_checkpoint1.pth.tar"))

# # train network
# for epoch in range(num_epoch):
#     losses = []
#     if epoch % 3 == 0:
#         checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
#         save_checkpoint(checkpoint)

#     for batch_idx, (data, targets) in enumerate(train_loader):
#         #get data to cuda if possible
#         data = data.to(device = device)
#         targets = targets.to(device = device)

#         #get to correct shape
#         # print(data.shape)
#         # data = data.reshape(data.shape[0], -1)

#         #forward
#         scores = model(data)
#         loss = criterion(scores, targets )

#         #backward
#         optimizer.zero_grad()
#         loss.backward()

#         #gradient descent or adam step
#         optimizer.step()
 




# # check accuracy on training and testing
# def check_accuracy(loader, model):
#     if loader.dataset.train:
#         print('checking accuracy onn training data')
#     else:
#         print('Checking accuracy on testing data')    
#     num_correct = 0
#     num_samples = 0
#     model.eval()

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device = device)
#             y = y.to(device = device)
#             # x = x.reshape(x.shape[0], -1)

#             scores = model(x)
#             _, predictions = scores.max(1)
#             num_correct += (predictions == y).sum()
#             num_samples += predictions.size(0)
        
        
#         print(
#             f"Got {num_correct} / {num_samples} with accuracy"
#             f" {float(num_correct) / float(num_samples) * 100:.2f}"
#         )
    
#     model.train()
#     # return acc

# check_accuracy(train_loader, model)
# check_accuracy(test_loader, model)



# Imports
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def main():
    # Initialize network
    model = torchvision.models.vgg16(
        weights=None
    )  # pretrained=False deprecated, use weights instead
    optimizer = optim.Adam(model.parameters())

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    # Try save checkpoint
    save_checkpoint(checkpoint)

    # Try load checkpoint
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


if __name__ == "__main__":
    main()