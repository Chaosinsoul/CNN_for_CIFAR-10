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


# In[3]:


net = vgg.vgg16(pretrained=True)
for param in net.parameters():
    param.requires_grad = False
net.classifier._modules['6'] = nn.Linear(4096, 256)


# In[4]:



net.cuda()

# Hyper parameters

optimizer = optim.Adam(net.classifier._modules['6'].parameters(), lr=learning_rate)
trainloader, validationloader, testloader = load_caltech256(batch_size)


# In[217]:


#Selecting the next batch
inputs = Variable(iter(trainloader).next()[0].cuda())


# In[158]:


#Showing an image in normal form (denormalized)
def show_image(image):
    for layer in image:
        layer += abs(layer.min())
    image = np.concatenate((image[0][:,:, np.newaxis], image[1][:, :,np.newaxis], image[2][:, :,np.newaxis]), axis=2)
    image = (image + abs(image.min())) / image.max()
    plt.imshow(image)


# In[159]:


#Shows a chosen number of activations/filtered images (for a selected image in the batch) for a spesific layer
def show_activations(inputs, layer_index, batch_index=0, num_images=float('inf')):
    for i in range(layer_index + 1):
        layer = net.features[i]
        inputs = layer(inputs)
    result = inputs.cpu().data.numpy()[batch_index]
    num_filters = min(len(result), num_images)
    plt.figure(figsize=(100,300))
    for filter in range(num_filters):
        rows = np.ceil(num_filters / 4)
        plt.subplot(rows, 4, filter + 1)
        plt.imshow(result[filter], cmap='gray')
    plt.show()

#Shows one chosen activation/filtered image (for a selected image in the batch) for a spesific layer
def show_activation(inputs, layer_index, filter_index=0, batch_index=0):
    for i in range(layer_index + 1):
        layer = net.features[i]
        inputs = layer(inputs)
    result = inputs.cpu().data.numpy()[batch_index]
    plt.imshow(result[filter_index], cmap='gray')


# In[183]:


#show_activations(inputs, layer_index=0, batch_index=0, num_images=40)


# In[161]:


#Shows a chosen number of weights for filters in a spesific layer
def show_weights(layer_index, num_images=float('inf')):
    weights = net.features[layer_index].weight.cpu().data.numpy()
    num_filters = min(len(weights), num_images)
    plt.figure(figsize=(100,300))
    maximum, minimum = np.max(weights), np.min(weights)
    for filter in range(num_filters):
        image = weights[filter]
        norm = (image + abs(minimum)) / maximum
        rows = np.ceil(num_filters / 4)
        plt.subplot(rows, 4, filter + 1)
        plt.imshow(norm)
    plt.show()

#Shows a spesific weight for a filter in a layer
def show_weight(layer_index, filter_index):
    weights = net.features[layer_index].weight.cpu().data.numpy()
    image = weights[filter_index]
    maximum, minimum = np.max(weights), np.min(weights)
    norm = (image + abs(minimum)) / maximum
    plt.imshow(norm)


# In[162]:


#show_weights(layer_index=0, num_images=40)


# In[163]:


def analize_filter(inputs, batch_index, layer_index, filter_index):
    plt.figure(figsize=(18, 54))
    plt.subplot(1, 3, 1)
    show_image(inputs.cpu().data.numpy()[batch_index])
    plt.subplot(1, 3, 2)
    show_activation(inputs, layer_index, filter_index, batch_index)
    plt.subplot(1, 3, 3)
    show_weight(layer_index, filter_index)
    plt.show()


# In[216]:


analize_filter(inputs, batch_index=0, layer_index=0, filter_index=13)
