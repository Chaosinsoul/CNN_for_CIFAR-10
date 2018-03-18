
# coding: utf-8

# In[1]:


from loader import load_caltech256
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tools import accuracy_score, compute_loss
from time import time
get_ipython().run_line_magic('matplotlib', 'inline')
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from caltech import Caltech256
from torchvision import datasets, transforms
import os
import torchvision.models.vgg as vgg
import numpy as np
from torchvision.transforms import ToPILImage


# In[2]:


class ProbingNet(nn.Module):
    def __init__(self):
        super(ProbingNet, self).__init__()
        net = vgg.vgg16(pretrained=True)
        self.block1 = nn.Sequential(net.features[0], net.features[1], net.features[2], net.features[3])
        self.block2 = nn.Sequential(net.features[4], net.features[5], net.features[6], net.features[7], net.features[8])
        self.block3 = nn.Sequential(net.features[9], net.features[10], net.features[11], net.features[12], net.features[13], net.features[14], net.features[15])
        self.block4 = nn.Sequential(net.features[16], net.features[17], net.features[18], net.features[19], net.features[20], net.features[21], net.features[22])
        self.pool = nn.MaxPool2d(2,2)
        self.fc= nn.Sequential(
            nn.Linear(28*28*256,1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )
        del net



    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
#        x = self.block4(x)
        x = x.view(-1, 28*28*256)
        x = self.fc(x)
        return x



# In[ ]:

# Full VGG net
'''
net = vgg.vgg16(pretrained=True)
full_net = False
net.cuda()
net = vgg.vgg16(pretrained=True)
for param in net.parameters():
    param.requires_grad = False

net.classifier._modules['6'] = nn.Linear(4096, 256)
'''

# In[ ]:


# Reduced net
net = ProbingNet()
net.cuda()

criterion = nn.CrossEntropyLoss()

# Hyper parameters
batch_size = 16
learning_rate = 0.000001
momentum = 0.9
early_stopping = 3
display_step = 1
# Optimizer
for param in net.parameters():
    param.requires_grad = False
for param in net.fc.parameters():
    param.requires_grad = True



params = [{'params': net.fc.parameters(), 'lr': learning_rate }]
optimizer = optim.Adam(params, lr=learning_rate)
trainloader, validationloader, testloader = load_caltech256(batch_size)


# In[ ]:


def tranform_labels(labels):
    labels = labels.type(torch.LongTensor).view(-1)
    # Transform from 1-256 to 0-255
    labels -= 1
    return labels
def accuracy_score(net, testloader, use_gpu=True):
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        labels = tranform_labels(labels)
        outputs = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        a = (predicted == labels.cuda()).sum()
        correct += a
    return correct / total

# Returns average loss for a criteria, loader and net
def compute_loss(criterion, loader, net, use_gpu=True):
    avg_loss = 0
    for images, labels in iter(loader):
        labels = tranform_labels(labels)
        images, labels = images.cuda(), labels.cuda()
        images, labels = Variable(images), Variable(labels)
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss = loss.data[0]
        avg_loss += loss
    return avg_loss / len(loader)


# In[ ]:


TEST_ACC = []
VAL_ACC = []
TRAIN_ACC = []
TEST_LOSS = []
VAL_LOSS = []
TRAIN_LOSS = []

start_time = time()
for epoch in range(50):  # loop over the dataset multiple times
    avg_loss = 0.0
    net.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        labels = tranform_labels(labels)
        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        avg_loss += loss.data[0]
        # Compute stats
    net.eval()

    if epoch % display_step == 0:
        avg_loss /= len(trainloader)
        val_loss = compute_loss(criterion, validationloader, net)

        VAL_LOSS.append(val_loss)
        test_acc = accuracy_score(net, testloader)

        val_acc = accuracy_score(net, validationloader)

        test_loss = compute_loss(criterion, testloader, net)

        train_acc = accuracy_score(net, trainloader)
        print("Epoch: {}, Time: {:.0f}, Train loss: {:.3f}, Validation accuracy: {:.3f}, Validation loss: {:.3f}".format(epoch, time() - start_time, avg_loss, val_acc, val_loss))

        TEST_ACC.append(test_acc)
        VAL_ACC.append(val_acc)
        TRAIN_ACC.append(train_acc)

        TEST_LOSS.append(test_loss)
        TRAIN_LOSS.append(avg_loss)
        avg_loss = 0
        if len(VAL_LOSS) > early_stopping:
            criterias = [VAL_LOSS[i] < VAL_LOSS[i+1] for i in range(len(VAL_LOSS)-early_stopping-1, len(VAL_LOSS)-1)]
            if sum(criterias) == early_stopping:
                break

print('Finished Training')


# ## Softmax probe
#
#

# In[6]:


plt.figure(figsize=(16,12))
plt.title("Train loss: {:.2f}, Test loss: {:.2f}, Val loss: {:.2f}, epochs: {}".format(TRAIN_LOSS[-early_stopping], TEST_LOSS[-early_stopping], VAL_LOSS[-early_stopping],epoch))
plt.xlabel("Epochs")
plt.ylabel("Cross entropy loss")
plt.plot(TEST_LOSS, label="Test loss")
plt.plot(VAL_LOSS, label="Validation loss")
plt.plot(TRAIN_LOSS, label="Training loss")
plt.legend()


# In[17]:


saved = [TEST_ACC, TEST_LOSS, TRAIN_ACC, TRAIN_LOSS, VAL_ACC, VAL_LOSS]


# In[7]:


plt.figure(figsize=(16,12))
plt.title("Train accuracy: {:.3f}, Test accuracy: {:.3f}, Val accuracy: {:.3f}, Epochs: {}".format(TRAIN_ACC[-early_stopping], TEST_ACC[-early_stopping], VAL_ACC[-early_stopping], epoch))
plt.xlabel("Epochs")
plt.ylabel("Classification accuracy")
plt.plot(TEST_ACC, label="Test accuracy")
plt.plot(VAL_ACC, label="Validation accuracy")
plt.plot(TRAIN_ACC, label="Training accuracy")
plt.legend()


# In[ ]:


def write_to_file(array, name):
    f = open(name, 'w')
    for a in array:
        f.write(str(a) + "\n")
    f.close()


# ##
