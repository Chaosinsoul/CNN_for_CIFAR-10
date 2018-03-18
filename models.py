from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Model2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=0),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=0),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5))            
        self.layer3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, padding=0),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1))
        self.layer4 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5))
        self.layer5 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1))
        self.layer6 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5))                
        self.fc1 = nn.Sequential(
            nn.Linear(3*3*192, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.25))
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.25))
        self.fc3 = nn.Sequential(
            nn.Linear(128, 10))
               
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(-1,3*3*192)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x



from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# cited from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.MaxPool2d(8)
#         self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
#         MaxPool2d(2)
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
       
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
net = ResNet(ResidualBlock, [2, 2, 2, 2])



class Model3(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128, 3, padding=1)
        
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,256,3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)        
        self.conv5 = nn.Conv2d(256,512,3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512,1024,3,padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        
        self.dropout = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(1024* 4 * 4, 10)
        
        nn.init.xavier_uniform(self.conv1.weight.data)
        nn.init.xavier_uniform(self.conv2.weight.data)
        nn.init.xavier_uniform(self.conv3.weight.data)
        nn.init.xavier_uniform(self.conv4.weight.data)
        nn.init.xavier_uniform(self.conv5.weight.data)
        nn.init.xavier_uniform(self.conv6.weight.data)  
        nn.init.xavier_uniform(self.fc1.weight.data)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = self.pool(F.leaky_relu(self.bn6(self.conv6(x))))
        x = self.dropout(x)
        
        
        x = x.view(-1, 1024 * 4 * 4)
        x = self.fc1(x)
        return x
