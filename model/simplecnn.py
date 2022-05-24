
import imp
import torch
import torch.nn as nn
from torchsummary import summary

class classifier_cnn(nn.Module):
    def __init__(self, num_classes):
        super(classifier_cnn, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 输出32*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出32*14*14  
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 64*14*14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64*7*7
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 64*7*7
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 64*3*3
        )
        self.fc = nn.Sequential(
            nn.Linear(64*3*3, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1) # 将feature变成一维向量
        x = self.fc(x)
        return x

class classifier_cnn2(nn.Module):
    def __init__(self, num_classes):
        super(classifier_cnn2, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3), # 输出32*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出32*14*14  
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # 64*14*14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64*7*7
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 64*7*7
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 64*3*3
        )
        self.fc = nn.Sequential(
            nn.Linear(64*3*3, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1) # 将feature变成一维向量
        x = self.fc(x)
        return x

class classifier_cnn3(nn.Module):
    def __init__(self, num_classes):
        super(classifier_cnn3, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2), # 输出32*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出32*14*14  
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 64*14*14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64*7*7
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 64*7*7
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 64*3*3
        )
        self.fc = nn.Sequential(
            nn.Linear(64*3*3, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1) # 将feature变成一维向量
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # test net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    model = classifier_cnn2(num_classes).to(device)
    img = torch.rand(1, 1, 28, 28).to(device)
    y = model(img)
    summary(model, input_size=(1,28,28))

    print(model)
    print(y)


