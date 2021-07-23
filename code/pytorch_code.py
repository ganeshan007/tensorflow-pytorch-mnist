import torch
import torch.utils.data.dataloader
import torchvision
import os
import numpy as np
from pprint import pprint
import sys
from tqdm.auto import tqdm
# sys.exit(0)

class Net(torch.nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_dim,out_dim)
        self.linear2 = torch.nn.Linear(out_dim,10)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self,x):
        x = x.view(10,-1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


dataset = torchvision.datasets.MNIST(root=r'/data',download=True,transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root=r'/data',train=False,transform=torchvision.transforms.ToTensor())
train_dataloader = torch.utils.data.dataloader.DataLoader(dataset=dataset,batch_size=10,num_workers=0)
test_dataloader = torch.utils.data.dataloader.DataLoader(dataset=test_dataset,batch_size=10,num_workers=0)

net = Net(784,400)
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(params=net.parameters(),lr=1e-2)
num_epochs = 10
for _ in range(num_epochs):
    epoch_loss = []
    test_epoch_loss = []
    for batch in tqdm(train_dataloader):
        net.train()
        x, y = batch[0], batch[1]
        y1 = net(x)
        loss = criterion(y1,y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        epoch_loss.append(loss.item())
    pprint(f'Epoch : {_}, Loss: {np.mean(epoch_loss)}')
    pprint('*'*50)

    net.eval()    
    for batch in test_dataloader:
        x, y = batch[0], batch[1]
        with torch.no_grad():
            y1 = net(x)
        loss = criterion(y1,y)
        test_epoch_loss.append(loss)
    pprint(f'Epoch {_}, Test Loss: {np.mean(test_epoch_loss)}')
    pprint('*'*50)

