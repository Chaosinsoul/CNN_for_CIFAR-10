
# coding: utf-8

# In[7]:


from loader import *
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


# In[8]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(), 
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.5),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(),   

        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.5),
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(1024, 2048, 3, padding=1),
            nn.BatchNorm2d(num_features=2048),
            nn.LeakyReLU(),

        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(2048, 4096, 3, padding=1),
            nn.BatchNorm2d(num_features=4096),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.5),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(4096*2*2, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),

        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

        )
        self.fc3 = nn.Linear(1024, 10)

        
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        
        x = x.view(-1, 4096 * 2 * 2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# In[9]:


net = Net()


# In[12]:


net.modules


# In[3]:


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)


# In[4]:


net = Net()
net.cuda()
net.apply(weights_init)
criterion = nn.CrossEntropyLoss()

# Hyper parameters
learning_rate = 0.0006
momentum = 0.9
batch_size = 16
early_stopping = 3
display_step = 1
# Optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


# In[ ]:


# Cuda settings: num_workers=1 and pin_memory=True
use_gpu = True
if use_gpu:
    num_workers = 1
    pin_memory = True
else:
    num_workers = 4
    pin_memory = False
trainloader, validationloder = get_train_valid_loader('./data', batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
testloader = get_test_loader('./data', batch_size=1000, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


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
    avg_loss /= len(trainloader)
    val_loss = compute_loss(criterion, validationloder, net)
    VAL_LOSS.append(val_loss)    
    if epoch % display_step == 0:    
        test_acc = accuracy_score(net, testloader)
        val_acc = accuracy_score(net, validationloder)
        test_loss = compute_loss(criterion, testloader, net)
        train_acc = accuracy_score(net, trainloader)
        print("Epoch: {}, Time: {:.0f}, Train loss: {:.3f}, Validation accuracy: {:.3f}, Validation loss: {:.3f}".format(epoch, time() - start_time, avg_loss, val_acc, val_loss))
            
        TEST_ACC.append(test_acc)
        VAL_ACC.append(val_acc)
        TRAIN_ACC.append(train_acc)

        TEST_LOSS.append(test_loss)
        TRAIN_LOSS.append(avg_loss)    
    if len(VAL_LOSS) > early_stopping:
        criterias = [VAL_LOSS[i] < VAL_LOSS[i+1] for i in range(len(VAL_LOSS)-early_stopping-1, len(VAL_LOSS)-1)]
        if sum(criterias) == early_stopping:
            break
print('Finished Training')


# In[ ]:


plt.figure(figsize=(16,12))
plt.title("Train loss: {:.2f}, Test loss: {:.2f}, Val loss: {:.2f}, epochs: {}".format(TRAIN_LOSS[-early_stopping], TEST_LOSS[-early_stopping], VAL_LOSS[-early_stopping],epoch))
plt.xlabel("Epochs")
plt.ylabel("Cross entropy loss")
plt.plot(TEST_LOSS, label="Test loss")
plt.plot(VAL_LOSS, label="Validation loss")
plt.plot(TRAIN_LOSS, label="Training loss")
plt.legend()


# In[ ]:


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

