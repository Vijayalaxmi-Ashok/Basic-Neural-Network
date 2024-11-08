import torch


#initializing the trensor
device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3],[4, 5, 6]], dtype = torch.float32, device = device, requires_grad = True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

#other common methods
x = torch.empty(size = (3, 3))
print(x)
y = torch.zeros((3, 3))
print(y)
z = torch.rand((4, 4))
print(z)
m = torch.ones((3, 3))
print(m)
n = torch.eye(5, 5)
print(n)
p = torch.arange(start = 0, end = 5, step = 1)
print(p)
q = torch.linspace(start = 0.1, end = 1,  steps = 10)
print(q)
