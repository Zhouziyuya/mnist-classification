import torch
from model.simplecnn import classifier_cnn, classifier_cnn2, classifier_cnn3
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# plot loss and accuracy
def plot_loss(loss1, loss2, loss3):
    plt.figure()
    loss1 = np.array(torch.tensor(loss1, device='cpu'))
    loss2 = np.array(torch.tensor(loss2, device='cpu'))
    loss3 = np.array(torch.tensor(loss3, device='cpu'))
    plt.plot(loss1, 'b-', label = '3 kernels with 3*3')
    plt.plot(loss2, 'r-', label = '3 kernels with 7*7 5*5 3*3')
    plt.plot(loss3, 'gold', label = '3 kernels with 5*5 3*3 3*3')
    plt.legend(fontsize=10)
    plt.title('train loss', fontsize = 10)
    plt.xlabel('Iteration')
    plt.savefig('./figures/loss.png')

def plot_accuracy(acc1, acc2, acc3):
    plt.figure()
    acc1 = np.array(acc1)
    acc2 = np.array(acc2)
    acc3 = np.array(acc3)
    plt.plot(acc1, 'b-', label = '3 kernels with 3*3')
    plt.plot(acc2, 'r-', label = '3 kernels with 7*7 5*5 3*3')
    plt.plot(acc3, 'gold', label = '3 kernels with 5*5 3*3 3*3')
    plt.legend(fontsize=10)
    plt.title('train accuracy', fontsize = 10)
    plt.xlabel('Iteration')
    plt.savefig('./figures/accuracy.png')

def train_one_epoch(loss_iter, accuracy_iter, model, data_loader, device, optimizer, loss_function, batch_size):
    model.train()
    n_correct = 0 # 每个epoch预测正确的总数
    n_total = 0 # 每个epoch预测总数

    for step, data in enumerate(data_loader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(torch.int64).to(device)

        output = model(imgs)
        err = loss_function(output, labels)
        loss_iter.append(err)

        optimizer.zero_grad()
        err.backward()
        optimizer.step()

        model.eval()
        test_output = model(imgs)
        _, indices = test_output.max(dim=1) # 找出行的最大值，返回索引
        n_correct += sum(indices==labels)
        n_total += batch_size
        
        acc_iter = sum(indices==labels).cpu().detach().numpy() * 1.0 / batch_size
        accuracy_iter.append(acc_iter)
    acc_epoch = n_correct.cpu().detach().numpy() * 1.0 / n_total # 计算每一个epoch的accuracy
        
    return loss_iter, accuracy_iter, acc_epoch



def main():

    # parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    lr = 1e-3
    num_classes = 10
    n_epoch = 2

    # preprocessing
    # normalize = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) # 图片归一化，三通道
    normalize = transforms.Normalize(mean=[0.5], std=[0.5]) # 图片归一化
    transform = transforms.Compose([transforms.ToTensor(), normalize]) # 转化为tensor向量然后归一化

    # download and load the data
    train_dataset = torchvision.datasets.MNIST(root='./dataset/', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./dataset/', train=False, transform=transform, download=False)

    # dataloader form
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # data = train_loader.dataset.data
    # shape = train_loader.dataset.data.shape # 查看dataloader的数据维度
    # print(shape)

    # load model, define optimizer and loss
    mynet = classifier_cnn(num_classes).to(device)
    optimizer = torch.optim.Adam(mynet.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss().to(device)

    # load model, define optimizer and loss
    mynet = classifier_cnn(num_classes).to(device)
    mynet2 = classifier_cnn2(num_classes).to(device)
    mynet3 = classifier_cnn3(num_classes).to(device)
    optimizer = torch.optim.Adam(mynet.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss().to(device)

    # train
    loss_value = []
    accuracy = []
    loss_value2 = []
    accuracy2 = []
    loss_value3 = []
    accuracy3 = []
    for epoch in range(n_epoch):

        loss_value, accuracy, acc_epoch = train_one_epoch(loss_value, accuracy, mynet, train_loader, device, optimizer, loss, batch_size)
        loss_value2, accuracy2, acc_epoch2 = train_one_epoch(loss_value2, accuracy2, mynet2, train_loader, device, optimizer, loss, batch_size)
        loss_value3, accuracy3, acc_epoch3 = train_one_epoch(loss_value3, accuracy3, mynet3, train_loader, device, optimizer, loss, batch_size)

        print(acc_epoch, acc_epoch2, acc_epoch3)  

    torch.save(mynet, './checkpoints/epoch50_3kernelsWith3x3.pth')
    torch.save(mynet2, './checkpoints/epoch50_3kernelsWith3x35x57x7.pth')
    plot_loss(loss_value, loss_value2, loss_value3)
    plot_accuracy(accuracy, accuracy2, accuracy3)

if __name__ == '__main__':
    main()
