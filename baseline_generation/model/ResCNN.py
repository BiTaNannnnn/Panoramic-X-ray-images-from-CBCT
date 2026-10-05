import torch
# 针对图像处理的包
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
# relu 激活函数
import torch.nn.functional as F
# 优化器的包
import torch.optim as optim
import matplotlib.pyplot as plt

#dataset
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ),(0.3081, ))
])

train_dataset = datasets.MNIST(root='xxx/dataset/mnist/',
                              train=True,
                              download=True,
                              transform = transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist/',
                              train=False,
                              download=True,
                              transform = transform)
test_loader = DataLoader(test_dataset,
                          shuffle=False,
                          batch_size=batch_size)

# Residual CNN
class ResidualCNN(torch.nn.Module):
    def __init__(self,channels):
        super(ResidualCNN, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels,channels,
                                     kernel_size=3,padding=1)
        self.conv2 = torch.nn.Conv2d(channels,channels,
                                     kernel_size=3,padding=1)

    def forward(self,x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        y = y + x
        y = F.relu(y)
        return y


# model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,16,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16,32,kernel_size=5)
        self.resblock_1 = ResidualCNN(16)
        self.resblock_2 = ResidualCNN(32)
        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(512,10)

    def forward(self,x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.resblock_1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.resblock_2(x)

        # tensor_len = x.shape[1] * x.shape[2] * x.shape[3]
        # print("最后输出张量 = ",tensor_len)
        x = x.view(in_size,-1)# flatten
        x = self.fc(x)
        return x

model = Net()
# loss + optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)


# train 每一轮循环封装到函数里
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader,0):
        inputs,target = data
        optimizer.zero_grad()

        #forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss = running_loss + loss.item()

        if batch_idx % 300 == 299:
            print("[%d,%5d] loss:%.3f" %(epoch+1,batch_idx+1,running_loss/300))
            running_loss =  0.0



def test():
    correct = 0
    total = 0
    global Accuracy_list

    # 不会计算梯度
    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            outputs = model(images)
            # 取每一行里最大数值的下标
            _, predicted = torch.max(outputs.data, dim = 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()

    correct1 = 100*correct/total
    Accuracy_list.append(correct1)
    print("Accuravy on test set:%d %%" % (100*correct/total))
    return Accuracy_list


if __name__ == '__main__':
    epoch_list = []
    Accuracy_list = []
    print("----")
    for epoch in range (10):
        train(epoch)
        test()
        epoch_list.append(epoch)

    print(epoch_list)
    print(Accuracy_list)
    plt.plot(epoch_list, Accuracy_list)
    plt.ylabel("Accuracy")
    plt.xlabel('epoch')
    plt.show()