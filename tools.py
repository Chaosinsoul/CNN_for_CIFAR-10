from torch.autograd import Variable
import torch

def accuracy_score(net, testloader, use_gpu=True):
    correct = 0
    total = 0
    # Set GPU options
    if use_gpu:
        net.cuda()
    else:
        net.cpu()
    for data in testloader:
        images, labels = data
        # Set GPU options
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()
        else:
            images = images.cpu()
            labels = labels.cpu()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        a = (predicted == labels).sum()
        correct += a
    return correct / total
# Returns average loss for a criteria, loader and net
def compute_loss(criterion, loader, net, use_gpu=True):
    avg_loss = 0
    for images, labels in iter(loader):
        if use_gpu:
            images, labels = images.cuda(), labels.cuda()
        else:
            images, labels = images.cpu(), labels.cpu()
        images, labels = Variable(images), Variable(labels)
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss = loss.data[0]
        avg_loss += loss
    return avg_loss / len(loader)